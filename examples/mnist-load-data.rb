# MNIST handwritten digits recognition

require 'redis'
require 'hiredis'

def insert_data(r,prefix,target,count)
    puts "Loading #{target} data..."
    image_filename = "mnist-data/#{prefix}-images-idx3-ubyte"
    label_filename = "mnist-data/#{prefix}-labels-idx1-ubyte"
    fi = File.open(image_filename)
    fl = File.open(label_filename)

    # Skip headers.
    fi.seek(16)
    fl.seek(8)
    # Load each char and respective label, sending it to the
    # neural network internal dataset.
    r.pipelined {
        (0...count).each{|i|
            bytes = fi.read(28*28).split("").map{|x| x.ord}
            label = fl.read(1).ord
            r.send('nr.observe',:mnist,*bytes,'->',label,target)
            puts "#{i+1}/#{count}" if (((i+1) % 5000) == 0)
        }
    }
end

r = Redis.new(:driver => :hiredis)
r.del(:mnist)
r.send('nr.create',:mnist,:classifier,28*28,100,'->',10,:DATASET,60000,:TEST,10000,:NORMALIZE)

insert_data(r,"train",:train,60000)
insert_data(r,"t10k",:test,10000)

puts "Start training with AUTOSTOP BACKTRACK for max 5000 cycles"

r.send('nr.train',:mnist,:maxtime,0,:maxcycles,500,:autostop,:backtrack)
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
        nn = r.send('nr.info',:mnist)
        nn = Hash[*nn]
        perc = 100.0 - nn['classification-errors-perc'].to_f
        puts "Best net so far can predict MNIST digits #{perc} of times"
        break
    end
end

