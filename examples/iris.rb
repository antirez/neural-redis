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
# One interesting feature of this dataset is that two of the classes are
# linearly separable, the another one is not.
#
# We use this as a benchmark of our auto-stop training.

require 'csv'
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

def train_iris(c,r)
    r.del(:iris)
    r.send('nr.create',:iris,:classifier,4,15,5,'->',3,:DATASET,1000,:TEST,500,:NORMALIZE)

    c.each{|x|
        id,a,b,c,d,iris_class = x
        r.send('nr.observe',:iris,a,b,c,d,'->',iris_class)
    }

    r.send('nr.train',:iris,:autostop,:backtrack,:maxtime,60000)
    wait_end_of_training(r)
    get_error(r)
end

c = CSV.open("Iris.csv").to_a
r = Redis.new

err = []
pass = 0
100.times {
    pass += 1
    err << train_iris(c,r)
    avgerr = err.reduce{|a,b| a+b}/err.length
    puts "#{pass} passes, this error: #{err[-1]}%, avg error: #{avgerr}%"
    break if err[-1] > 30
}
