# Hello world regressor

require 'redis'

r = Redis.new
r.del(:sumnet)
r.send('nr.create',:sumnet,:regressor,2,10,'->',1,:DATASET,1000,:TEST,500,:NORMALIZE)

150.times {
    a = rand(100)
    b = rand(100)
    r.send('nr.observe',:sumnet,a,b,'->',a+b)
}

# Also train with smaller numbers, since the above training
# set will be unbalanced torward bigger numbers.
150.times {
    a = rand(10)
    b = rand(10)
    r.send('nr.observe',:sumnet,a,b,'->',a+b)
}

r.send('nr.train',:sumnet,:maxtime,2000)
sleep(3)

puts "50 + 100 = #{r.send('nr.run',:sumnet,50,100)}"
puts "20 + 40 = #{r.send('nr.run',:sumnet,20,40)}"
puts "2 + 4 = #{r.send('nr.run',:sumnet,2,4)}"
