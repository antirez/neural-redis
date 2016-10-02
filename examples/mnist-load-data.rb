# MNIST handwritten digits recognition

require 'redis'

def get_error(r)
    a = r.send('nr.info',:iris)
    Hash[*a]["classification-errors-perc"].to_f
end

def is_training(r)
    a = r.send('nr.info',:iris)
    Hash[*a]["training"].to_i == 1
end

def wait_end_of_training(r)
    while is_training(r)
        sleep 0.1
    end
end

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

r = Redis.new
r.del(:mnist)
r.send('nr.create',:mnist,:classifier,28*28,100,'->',10,:DATASET,60000,:TEST,10000,:NORMALIZE)

insert_data(r,"train",:train,60000)
insert_data(r,"t10k",:test,10000)

puts "Data loading finished. You can now train the NN with:\n"
puts "  NR.TRAIN mnist MAXCYCLES 20 MAXTIME 0\n"
puts "Note: 20 cycles are not enough with 100 hidden units, check the"
puts "classifications error with NR.INFO and start new trainings to see"
puts "how it changes over time."
