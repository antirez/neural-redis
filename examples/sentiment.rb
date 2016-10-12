# 2000 movies sentiment analysis
#
# Copyright (C) 2016 Salvatore Sanfilippo
# All Rights Reserved
#
# Licensed under BSD two clause license, se COPYING file in this
# software distribution.

SentimentNeg = 0
SentimentPos = 1

require 'redis'
require 'hiredis'
require 'zlib'

# Just return a list of sentences: the source files are already organized
# with a sentence per line.
def get_sentences(filename)
    sentences = File.open(filename).to_a
    sentences
end

# We need to map a sentence to a fixed set of inputs in our neural network.
# Here I did the most crude thing that could work at all... not preprocessing
# the sentence in any way. I've NumInputs (a few thousands, it's configurable)
# We split the inputs in two sides: singe word inputs, successive word inputs.
#
# Then we find the position of the two inputs to set:
#
#   POS1 = HASH(Word) MOD NumInputs
#   POS2 = HASH(Word+Next_Word) MOD NumInputs
#
# We do that for all the words (and sequences of two words) and finally
# normalize so that the sum of all the inputs set is 1.
#
# This trivial approach works, getting around 80% with 3000 inputs and
# 150 hidden layers.

NumInputs = 3003
NumSections = 2

def sentences_to_inputs(sentences)
    iv = [0]*NumInputs
    sum = 0
    sentences.each{|s|
        inputs = s.gsub(/[^a-z,! ]/," ").gsub(/ +/," ")
        inputs = inputs.split(" ")

        # Section 1: each single word
        inputs.each{|w|
            h = Zlib.crc32(w) % (NumInputs/NumSections)
            iv[h] = iv[h]+1
            sum += 1
        }

        # Section 2: each word and the next one
        (0...(inputs.length-1)).each{|i|
            w1 = inputs[i]
            w2 = inputs[i+1]
            next if w2 == "," || w2 == "!"
            h = Zlib.crc32(w1+"."+w2) % (NumInputs/NumSections)
            h += NumInputs/NumSections
            iv[h] = iv[h]+1
            sum += 1
        }
    }
    sum = 1 if sum == 0
    iv.map{|x| x.to_f/sum}
end

# Insert the 2000 sentences. We split 1600 / 400 (training / testing).
# We make it up to Redis to select what items put in what dataset.
# The number of items are setup when we create the NN.
def insert_data(r,dirname,sentiment)
    puts "Loading #{dirname} data..."
    files = Dir.open(dirname).to_a
    r.pipelined {
        i = 0
        files.each{|f|
            next if f == "." || f == ".."
            sentences = get_sentences(dirname+f)
            inputs = sentences_to_inputs(sentences)
            r.send('nr.observe',:sentiment,*inputs,'->',sentiment)
            puts "#{i+1}/#{files.length}" if (((i+1) % 100) == 0)
            i += 1
        }
    }
end

# Test function, not used.
def test_it(r,filename,expected)
    files = Dir.open(filename).to_a
    errors = 0
    files.each{|f|
        next if f == "." || f == ".."
        sentences = get_sentences(filename+f)
        inputs = sentences_to_inputs(sentences)
        outputs = r.send('nr.run',:sentiment,*inputs)
        oclass = r.send('nr.class',:sentiment,*inputs).to_i
        errors += 1 if (oclass != expected)
    }
    puts "Errors: #{errors}/#{files.length}"
end

# Let users have some fun.
def interactive(r)
    puts "Imagine and type a film review sentence:"
    while true
        print "\n> "
        STDOUT.flush
        s = STDIN.gets
        inputs = sentences_to_inputs(s.split("."))
        outputs = r.send('nr.run',:sentiment,*inputs)
        puts "Negativity: #{outputs[SentimentNeg]}"
        puts "Positivity: #{outputs[SentimentPos]}"
    end
end

r = Redis.new(:driver => :hiredis)
r.del(:sentiment)

r.send('nr.create',:sentiment,:classifier,NumInputs,51,'->',2,:DATASET,1400,:TEST,600)

insert_data(r,"sentiment/txt_sentoken/neg/",SentimentNeg)
insert_data(r,"sentiment/txt_sentoken/pos/",SentimentPos)

# Train the network, and when it's done, show the percentage
# of accuracy.

puts "Start training with AUTOSTOP BACKTRACK for max 50 cycles"

r.send('nr.train',:sentiment,:maxtime,0,:maxcycles,100,:autostop,:backtrack)
oldinfo = nil

start=Time.now
while true
    info = r.send('nr.threads')
    if (info.length != 0 && info != oldinfo)
        timeinfo = " milliseconds_per_cycle=#{(Time.now-start)*1000}"
        start = Time.now
        puts info[0] + timeinfo
        oldinfo = info
    end
    sleep 0.01
    if info.length == 0
        puts ""
        nn = r.send('nr.info',:sentiment)
        nn = Hash[*nn]
        perc = 100.0 - nn['classification-errors-perc'].to_f
        puts "Best net so far can predict sentiment polarity #{perc} of times"
        break
    end
end

#test_it(r,"sentiment/txt_sentoken/neg/",SentimentNeg)
#test_it(r,"sentiment/txt_sentoken/pos/",SentimentPos)
interactive(r)
