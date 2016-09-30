# This is an example with a very popular dataset that classifies
# Iris in three different sub-species:
#
#   Type 0: Iris setosa
#   Type 1: Iris versicolor
#   Type 2: Iris virginica
#
# There CSV is composed of id,feature_a,feature_b,feature_c,feature_d,iris_type
#
# The features are the sepal length and width and the petal length and width.

require 'csv'
require 'redis'

c = CSV.open("Iris.csv").to_a

r = Redis.new
r.del(:iris)
r.send('nr.create',:iris,:classifier,4,15,'->',3,:DATASET,1000,:TEST,500,:NORMALIZE)

c.each{|x|
    id,a,b,c,d,iris_class = x
    r.send('nr.observe',:iris,a,b,c,d,'->',iris_class)
}

r.send('nr.train',:iris,:autostop,:maxtime,60000)
while true
    nninfo = r.send('nr.info',:iris)
    break if nninfo[7] == 0
    puts "Still training..."
    sleep(1)
end
puts r.send('nr.info',:iris)
