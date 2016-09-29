require 'csv'
require 'redis'

# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
c = CSV.open("titanic.csv").to_a

# Create the neural network. Inputs are: Pclass, Sex, Age, SibSp, Parch, Fare.
# However class and sex will use an input for each different case, so we
# need a total of 9 inputs. The outputs will be just two: surived, not survived.
r = Redis.new
r.del(:mynet)
r.send('nr.create',:mynet,:classifier,9,15,'->',2,:DATASET,1000,:TEST,500,:NORMALIZE)

dataset = []
c.each{|x|
    passid,survival,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked = x
    # The average age in the dataset is about 30 years old, we use this
    # to avoid discarding too many items
    age = 30 if !age
    dataset << {:pclass => pclass.to_i,
                :female => (sex == "female") ? 1 : 0,
                :male => (sex == "male") ? 1 : 0,
                :age => age.to_f,
                :sibsp => sibsp.to_f,
                :parch => parch.to_f,
                :fare => fare.to_f,
                :survived => (survival == "1") ? 1 : 0}
}

def feed_data(r,dataset,mode)
    errors = 0
    dataset.each{|d|
        pclass = [0,0,0]
        pclass[d[:pclass]-1] = 1
        inputs = pclass +
                 [d[:male],d[:female]] +
                 [d[:age],
                  d[:sibsp],
                  d[:parch],
                  d[:fare]]
        outputs = d[:survived]
        if mode == :observe
            r.send('nr.observe',:mynet,*inputs,'->',outputs)
        elsif mode == :test
            res = r.send('nr.class',:mynet,*inputs)
            if res != outputs
                errors += 1
            else
                print "#{inputs} -> #{outputs}\n"
            end
        end
    }
    if mode == :test
        puts "#{errors} prediction errors on #{dataset.length} items"
    end
end

training_dataset = dataset[0..600]
test_dataset = dataset[601..-1]

puts "Before training"
feed_data(r,test_dataset,:test)

feed_data(r,training_dataset,:observe)
r.send('nr.train',:mynet,:autostop,:maxtime,60000)
while true
    nninfo = r.send('nr.info',:mynet)
    break if nninfo[7] == 0
    puts "Still training..."
    sleep(1)
end
puts "After training"
feed_data(r,test_dataset,:test)
