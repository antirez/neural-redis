require 'redis'

r = Redis.new
r.del(:mynet2)
r.send('nr.create',:mynet2,:classifier,2,20,'->',3,:DATASET,1000,:TEST,100)

def feed_dataset(r,count,mode)
    k = 0.0
    errors = 0
    count.times {
        point_class = rand(3)
        if point_class == 0
            x = Math.sin(k)/2+rand()/10;
            y = Math.cos(k)/2+rand()/10;
        elsif point_class == 1
            x = Math.sin(k)/3+0.4+rand()/8;
            y = Math.cos(k)/4+0.4+rand()/6;
        else
            x = Math.sin(k)/3-0.5+rand()/30;
            y = Math.cos(k)/3+rand()/40;
        end
        if mode == :feed
            r.send('nr.observe',:mynet2,x,y,'->',point_class)
        elsif mode == :check
            res = r.send('nr.class',:mynet2,x,y)
            if point_class != res
                errors += 1
            end
        end
        k += Math::PI*2/count
    }
    return errors
end

def print_map(r)
    (0...80).each{|y|
        (0...80).each{|x|
            res = r.send('nr.class',:mynet2,(x-40.0)/40,(y-40.0)/40)
            charset = ".O/"
            print(charset[res])
        }
        puts
    }
end

feed_dataset(r,1000,:feed)
puts "Data sent, training"
r.send('nr.train',:mynet2,:autostop,:backtrack,:maxtime,5000)

while true
    nninfo = r.send('nr.info',:mynet2)
    break if nninfo[7] == 0
    puts "Still training..."
    sleep(1)
end

puts "Testing"
print_map(r)
puts "Erors: #{feed_dataset(r,1000,:check)}/1000"
puts r.send('nr.info',:mynet2)
