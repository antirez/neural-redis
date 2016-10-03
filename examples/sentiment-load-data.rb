# 2000 movies sentiment analysis

SentimentNeg = 0
SentimentPos = 1
NumInputs = 3000
NumSections = 2

require 'redis'
require 'hiredis'
require 'zlib'

def get_sentences(filename)
    sentences = File.open(filename).to_a
    sentences
end

def sentences_to_inputs(sentences)
    iv = [0]*NumInputs
    sentences.each{|s|
        inputs = s.gsub(/[^a-z,! ]/," ").gsub(/ +/," ")
        inputs = inputs.split(" ")

        # Section 1: each single word
        inputs.each{|w|
            h = Zlib.crc32(w) % (NumInputs/NumSections)
            iv[h] = iv[h]+1
        }

        # Section 2: each word and the next one
        (0...(inputs.length-1)).each{|i|
            w1 = inputs[i]
            w2 = inputs[i+1]
            next if w2 == "," || w2 == "!"
            h = Zlib.crc32(w1+"."+w2) % (NumInputs/NumSections)
            h += NumInputs/NumSections
            iv[h] = iv[h]+1
        }
    }
    iv
end

def insert_data(r,dirname,sentiment)
    puts "Loading #{dirname} data..."
    files = Dir.open(dirname).to_a
    puts r.pipelined {
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
#        puts "Negativity: #{outputs[SentimentNeg]}"
#        puts "Positivity: #{outputs[SentimentPos]}"
    }
    puts "Errors: #{errors}/#{files.length}"
end

def interactive(r)
    while true
        puts "Imagine and type a film review sentence: "
        s = STDIN.gets
        inputs = sentences_to_inputs(s.split("."))
        outputs = r.send('nr.run',:sentiment,*inputs)
        puts "Negativity: #{outputs[SentimentNeg]}"
        puts "Positivity: #{outputs[SentimentPos]}"
    end
end

r = Redis.new(:driver => :hiredis)
test_it(r,"sentiment/txt_sentoken/neg/",SentimentNeg)
test_it(r,"sentiment/txt_sentoken/pos/",SentimentPos)
interactive(r)
exit

r.del(:sentiment)

r.send('nr.create',:sentiment,:classifier,NumInputs,100,'->',2,:DATASET,1400,:TEST,600,:NORMALIZE)

insert_data(r,"sentiment/txt_sentoken/neg/",SentimentNeg)
insert_data(r,"sentiment/txt_sentoken/pos/",SentimentPos)

puts "Start training with AUTOSTOP BACKTRACK for max 50 cycles"

r.send('nr.train',:sentiment,:maxtime,0,:maxcycles,50,:autostop,:backtrack)
oldinfo = nil
while true
    info = r.send('nr.threads')
    if (info != oldinfo)
        puts info
        oldinfo = info
    end
    sleep 0.1
    if info.length == 0
        puts ""
        nn = r.send('nr.info',:sentiment)
        nn = Hash[*nn]
        perc = 100.0 - nn['classification-errors-perc'].to_f
        puts "Best net so far can predict sentiment polarity #{perc} of times"
        break
    end
end


