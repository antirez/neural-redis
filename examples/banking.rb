# This example is the same used by Amazon ML service as tutorial form.
#
# There CSV is composed of a set of customer features and the way the
# customer was contacted, and if the customer actually subscribed to a
# service or not. Our goal is to predict if customers will subscribe.

require 'csv'
require 'redis'
require 'hiredis'

$classes = {}

def class_to_class_id(class_type,class_str)
    $classes[class_type] = {} if !$classes[class_type]
    $classes[class_type][class_str] = $classes[class_type].length if !$classes[class_type][class_str]
    class_id = $classes[class_type][class_str]
    return class_id
end

def class_to_inputs(class_type,class_str,numclasses)
    class_id = class_to_class_id(class_type,class_str)
    array = [0]*numclasses
    array[class_id] = 1
    return array
end

def csvrow_to_net(row)
    age,job,marital,education,default,housing,loan,contact,month,day_of_week,duration,campaign,pdays,previous,poutcome,emp_var_rate,cons_price_idx,cons_conf_idx,euribor3m,nr_employed,y = row
    input = [age]
    input += class_to_inputs(:job,job,12)
    input += class_to_inputs(:marital,marital,4)
    input += class_to_inputs(:education,education,8)
    input += class_to_inputs(:default,default,3)
    input += class_to_inputs(:housing,housing,3)
    input += class_to_inputs(:load,loan,3)
    input += class_to_inputs(:contact,contact,2)
    input += class_to_inputs(:month,month,12)
    input += class_to_inputs(:day_of_week,day_of_week,5)
    input += [campaign]
    input += [pdays]
    input += [previous]
    input += class_to_inputs(:poutcome,poutcome,3)
    input += [cons_price_idx]
    input += [cons_conf_idx]
    input += [euribor3m]
    input += [nr_employed]
    [input,class_to_class_id(:y,y)]
end

def load_banking(c,r)
    r.del(:banking)
    r.pipelined {
        c.each_with_index{|row,i|
            next if i == 0 # First row of CSV is just labels name
            inputs,outputs = csvrow_to_net(row)
            if i == 1
                r.send('nr.create',:banking,:classifier,inputs.length,10,'->',2,:DATASET,70000,:TEST,30000,:NORMALIZE)
            end

            r.send('nr.observe',:banking,*inputs,'->',*outputs)
        }
    }
end

c = CSV.open("banking.csv").to_a
r = Redis.new(:driver => :hiredis)

puts "Loading data"

load_banking(c,r)

puts "Start training with AUTOSTOP BACKTRACK for max 5000 cycles"

r.send('nr.train',:banking,:maxtime,0,:maxcycles,500,:autostop,:backtrack)
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
        nn = r.send('nr.info',:banking)
        nn = Hash[*nn]
        perc = 100.0 - nn['classification-errors-perc'].to_f
        puts "Best net so far can predict outcome #{perc} of times"
        break
    end
end

