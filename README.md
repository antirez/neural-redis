Neural Redis
===

*Machine learning is like highschool sex. Everyone says they do it, nobody really does, and no one knows what it actually is.* -- [@Mikettownsend](https://twitter.com/Mikettownsend/status/780453119238955008).

Neural Redis is a Redis loadable module that implements feed forward neural
networks as a native data type for Redis. The project goal is to provide
Redis users with an extremely simple to use machine learning experience.

Normally machine learning is operated by collecting data, training some
system, and finally executing the resulting program in order to solve actual
problems. In Neural Redis all this phases are compressed into a single API:
the data collection and training all happen inside the Redis server.
Neural networks can be executed while there is an ongoing training, and can
be re-trained multiple times as new data from the outside is collected
(for instance user events).

The project starts from the observation that, while complex problems like
computer vision need slow to train and complex neural networks setups, many
regression and classification problems that are able to enhance the user
experience in many applications, are approachable by feed forward fully
connected small networks, that are very fast to train, very generic, and
robust against non optimal parameters configurations.

Neural Redis implements:

* A very simple to use API.
* Automatic data normalization.
* Online training of neural networks in different threads.
* Ability to use the neural network while the system is training it (we train a copy and only later merge the weights).
* Fully connected neural networks using the RPROP (Resilient back propagation) learning algorithm.
* Automatic training with simple overtraining detection.

The goal is to help developers, especially of mobile and web applications, to
have a simple access to machine learning, in order to answer questions like:

* What promotion could work most likely with this user?
* What AD should I display to obtain the best conversion?
* What template is the user likely to appreciate?
* What is a likely future trend for this data points?

Of course you can do more, since neural networks are pretty flexible. You
can even have fun with computer visions datasets like
[MINST](http://yann.lecun.com/exdb/mnist/), however keep in mind that
the neural networks implemented in Neural Redis are not optimized for
complex computer visions tasks like convolutional networks (it will
score 2.3%, very far from the state of art!), nor Neural Redis implements
the wonders of recurrent neural networks.

However you'll be surpirsed by the number of tasks in which a simple
neural network that can be trained in minutes, will be able to discover
linear ad non linear correlations.

Loading the extension
===

To run this extension you need Redis `unstable`, grab it from Github, it
is the default branch. Then compile the extension, and load it starting
Redis with:

    redis-server --loadmodule /path/to/neuralredis.so

Alternatively add the following in your `redis.conf` file:

    loadmodule /path/to/neuralredis.so

WARNING: alpha code
===

**WARNING:** this is alpha code. It is likely to contain bugs and may
easily crash the Redis server. Also note that currently only
RDB persistence is implemented in the module, while AOF rewrite
is not implemented at all. Use at your own risk.

If you are not still scared enough, please consider that I wrote the
more than 1000 lines of C code composing this extension, and this
README file, in roughly two days.

Note that this implementation may be hugely improved. For instance
currently only the sigmoid activaction function and the root mean
square loss functions are supported: while for the problems this
module is willing to address this limited neural network implementation
is showing to be quite flexible, it is possible to do much better
depending on the problem at hand.

Hello World
===

In order to understand how the API works, here is an hello world example
where we'll teach our neural network to do... additions :-)

To create a new neural network we use the following command:

    > NR.CREATE net REGRESSOR 2 3 -> 1 NORMALIZE DATASET 50 TEST 10
    (integer) 13

The command creates a neural network, configured for regression tasks
(as opposed to classification: well'll explain what this means
in the course of this tutorial).

Note that the command replied with "13". It means that the network
has a total of 13 tunable parameters, considering all the weights
that go from units or biases to other units. Larger networks
will have a lot more parameters.

The neural network has 2 inputs, 3 hidden layers, and a single output.
Regression means that given certain inputs and desired outputs, we want the
neural network to be able to *understand* the function that given the
inputs computes the outputs, and compute this function when new inputs
are presented to it.

The `NORMALIZE` option means that it is up to Redis to normalize the
data it receives, so there is no need to provide data in the -/+ 1 range.
The options `DATASET 50` and `TEST 10` means that we want an internal
memory for the dataset of 50 and 10 items respectively for the training
dataset, and the testing dataset.

The learning happens using the training dataset, while the testing dataset
is used in order to detect if the network is able to generalize, that is,
is really able to understand how to approximate a given function.
At the same time, the testing dataset is useful to avoid to train the network
too much, a problem known as *overfitting*. Overfitting means that the
network becomes too much specific, at the point to be only capable of replying
correctly to the inputs and outputs it was presented with.

Now it is time to provide the network with some data, so that it can learn
the function we want to approximate:

    > NR.OBSERVE net 1 2 -> 3
    1) (integer) 1
    2) (integer) 0

We are saying: given the inputs 1 and 2, the output is 3.
The reply to the `NR.OBSERVE` command is the number of data items
stored in the neural network memory, respectively in the training
and testing data sets.

We continue like that with other examples:

    > NR.OBSERVE net 4 5 -> 9
    > NR.OBSERVE net 3 4 -> 7
    > NR.OBSERVE net 1 1 -> 2
    > NR.OBSERVE net 2 2 -> 4
    > NR.OBSERVE net 0 9 -> 9
    > NR.OBSERVE net 7 5 -> 12
    > NR.OBSERVE net 3 1 -> 4
    > NR.OBSERVE net 5 6 -> 11

At this point we need to train the neural network, so that it
can learn:

    > NR.TRAIN net AUTOSTOP

The `NR.TRAIN` command starts a training thread. the `AUTOSTOP` option
means that we want the training to stop before overfitting starts
to happen.

Using the `NR.INFO` command you can see if the network is still training.
However in this specific case, the network will take a few milliseconds to
train, so we can immediately try if it actually learned how to add two
numbers:

    > NR.RUN net 1 1
    1) "2.0776522297040843"

    > NR.RUN net 3 2
    1) "5.1765427204933099"

Well, more or less it works. Let's look at some internal info now:

    > NR.INFO net
     1) id
     2) (integer) 1
     3) type
     4) regressor
     5) auto-normalization
     6) (integer) 1
     7) training
     8) (integer) 0
     9) layout
    10) 1) (integer) 2
        2) (integer) 3
        3) (integer) 1
    11) training-dataset-maxlen
    12) (integer) 50
    13) training-dataset-len
    14) (integer) 6
    15) test-dataset-maxlen
    16) (integer) 10
    17) test-dataset-len
    18) (integer) 2
    19) training-total-steps
    20) (integer) 1344
    21) training-total-seconds
    22) 0.00
    23) dataset-error
    24) "7.5369825612397299e-05"
    25) test-error
    26) "0.00042670663615723583"
    27) classification-errors-perc
    28) 0.00

As you can see we have 6 dataset items and 2 test items. We configured
the network at creation time to have space for 50 and 10 items. As you add
items with `NR.OBSERVE` the network will put items evenly on both datasets,
proportionally to their respective size. Finally when the datasets are full,
old random entries are replaced with new
ones.

We can also see that the network was trained with 1344 steps for 0 seconds
(just a few milliseconds). Each step is the training performed with a single
data item, so the same 6 items were presented to the network for 244 cycles
in total.

A few words about normalization
===

If we try to use our network with values outside the range it learned with,
we'll see it failing:

    > NR.RUN net 10 10
    1) "12.855978185382257"

This happens because the automatic normalization will consider the maximum
values seen in the training dataset. So if you plan to use auto normalization,
make sure to show the network samples with different values, including inputs
at the maximum of the data you'll want to use the network with in the future.

Classification tasks
===

Regression approximates a function having certain inputs and outputs in the
training data set. Classification instead is the task of, given a set of
inputs representing *something*, to label it with one of a fixed set of
labels.

For example the inputs may be features of Greek jars, and the classification
output could be one of the following three jar types:

* Type 0: Kylix type A
* Type 1: Kylix type B
* Type 2: Kassel cup

As a programmer you may think that, the output class, is just a single output
number. However neural networks don't work well this way, for example
classifying type 0 with an output between 0 and 0.33, type 1 with an output
between 0.33 and 0.66, and finally type 2 with an output between 0.66 and 1,
will not work well at all.

The way to go instead is to use three distinct outputs, where we set two
always to 0, and a single one to 1, corresponding to the type the output
represents, so:

* Type 0: [1, 0, 0]
* Type 1: [0, 1, 0]
* Type 2: [0, 0, 1]

When you create a neural network with the `NR.CREATE` command, and use as
second argument `CLASSIFIER` instead of `REGRESSOR`, Neural Redis will do
the above transformation for you, so when you train your network with
`NR.OBSERVE` you'll just use, as output, as single number: 0, 1 or 2.

Of course, you need to create the network with three outputs like that:

    > NR.CREATE mynet CLASSIFIER 5 10 -> 3
    (integer) 93

Our network is currently untrained, but it can already be run, even if the
replies it will provide are totally random:

    > NR.RUN mynet 0 1 1 0 1
    1) "0.50930603602918945"
    2) "0.48879876200255651"
    3) "0.49534453421381375"

As you can see, the network *voted* for type 0, since the first output is
greater than the others. There is a Neural Redis command that saves you the
work of finding the greatest output client side in order to interpret the
result as a number between 0 and 2. It is identical to `NR.RUN` but just
outputs directly the class ID, and is called `NR.CLASS`:

    > NR.CLASS mynet 0 1 1 0 1
    (integer) 0

However note that ofter `NR.RUN` is useful for classification problems.
For example a blogging platform may want to train a neural network to
predict the template that will appeal more to the user, based on the
registration data we just obtained, that include the country, sex, age
and category of the blog.

While the prediction of the network will be the output with the highest
value, if we want to present different templates, it makes sense to
present, in the listing, as the second one the one with the second
highest output value and so forth.

Before diving into a practical classification example, there is a last
thing to say. Networks of type CLASSIFIER are also trained in a different
way: instead of giving as output a list of zeros and ones you directly
provide to `NR.OBSERVE` the class ID as a number, so in the example
of the jars, we don't need to write `NR.OBSERVE 1 0.4 .2 0 1 -> 0 0 1` to
specify as output of the provided data sample the third class, but
we should just write:

    > NR.OBSERVE mynet 1 0.4 .2 0 1 -> 2

The "2" will be translated into "0 0 1" automatically, as "1"
would be translated to "0 1 0" and so forth.

A practical example: the Titanic dataset
===

[Kaggle.com](https://www.kaggle.com/) is hosting a machine learning
competition. One of the datasets they use, is the list of the Titanic
passengers, their ticket class, fair, number of relatives, age,
sex, and other information, and... If they survived or not during the
Titanic incident.

You can find both the code and a CSV with a reduced dataset of 891
entries in the `examples` directory of this Github repository.

In this example we are going to try to predict, given a few input
variables, if a specific person is going to survive or not, so this
is a classification task, where we label persons with two different
labels: survived or died.

This problem is pretty similar, even if a bit more scaring, than the
problem of labeling users or predicting their response in some web
application according to their behavior and the other data we collected
in the past (hint: machine learning is all about collecting data...).

In the CSV there are a number of information about each passenger,
but here in order to make the example simpler we'll use just the
following fields:

* Ticket class (1st, 2nd, 3rd).
* Sex.
* Age.
* Sibsp (Number of siblings, spouses aboard).
* Parch (Number of parents and children aboard).
* Ticket fare.

If there is a correlation between this input variables and the
ability to survive, our neural network should find it.

Note that while we have six inputs, we'll need a total network
with 9 total inputs, since sex and ticket class, are actually
*input classes*, so like we did in the output, we'll need to
do in the input. Each input will signal if the passenger is in
one of the possible classes. This are our nine inputs:

* Is male? (0 or 1).
* Is Female? (0 or 1).
* Traveled in first class? (0 or 1).
* Traveled in second class? (0 or 1).
* Traveled in third class? (0 or 1).
* Age.
* Number of siblings / spouses.
* Number of parents / children.
* Ticket fare.

We have a bit less than 900 passengers (I'm using a reduced
dataset here), however we want to take about 200 for verification
at application side, without sending them to Redis at all.

The neural network will also use part of the dataset for
verification, since here I'm planning to use the automatic training
stop feature, in order to detect overfitting.

Such a network can be created with:

    > NR.CREATE mynet CLASSIFIER 9 15 -> 2 DATASET 1000 TEST 500 NORMALIZE

Also note that we are using a neural network with a single hidden
layer (the layers between inputs and outputs are called hidden, in
case you are new to neural networks). The hidden layer has 15 units.
This is still a pretty small network, but we expect that for the
amount of data and the kind of correlations that there could be in
this data, this could be enough. It's possible to test with
different parameters, and I plan to implement a `NR.CONFIGURE`
command so that it will be possible to change this things on the fly.

Also note that since we defined a testing dataset maximum size to be half
the one of the training dataset (1000 vs 500), `NR.OBSERVE` will automatically
put one third of the entires in the testing dataset.

If you check the Ruby program that implements this example inside the
source distribution, you'll see how data is fed directly as it is
to the network, since we asked for auto normalization:

```
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
            end
        end
    }
    if mode == :test
        puts "#{errors} prediction errors on #{dataset.length} items"
    end
end
```

The function is able to both send data or evaluate the error rate.

After we load 601 entries from the dataset, before any training, the output
of `NR.INFO` will look like this:

    > NR.INFO mynet
     1) id
     2) (integer) 5
     3) type
     4) classifier
     5) auto-normalization
     6) (integer) 1
     7) training
     8) (integer) 0
     9) layout
    10) 1) (integer) 9
        2) (integer) 15
        3) (integer) 2
    11) training-dataset-maxlen
    12) (integer) 1000
    13) training-dataset-len
    14) (integer) 401
    15) test-dataset-maxlen
    16) (integer) 500
    17) test-dataset-len
    18) (integer) 200
    19) training-total-steps
    20) (integer) 0
    21) training-total-seconds
    22) 0.00
    23) dataset-error
    24) "0"
    25) test-error
    26) "0"
    27) classification-errors-perc
    28) 0.00
    29) overfitting-detected
    30) no

As expected, we have 401 training items and 200 testing dataset.
Note that for networks declared as classifiers, we have an additional
field in the info output, which is `classification-errors-perc`. Once
we train the network this field will be populated with the percentage (from
0% to 100%) of items in the testing dataset which were misclassified by
the neural network. It's time to train our network:

    > NR.TRAIN mynet AUTOSTOP
    Training has started

If we check the `NR.INFO` output after the training, we'll discover a few
interesting things (only quoting the relevant part of the output):

    19) training-total-steps
    20) (integer) 64160
    21) training-total-seconds
    22) 0.29
    23) dataset-error
    24) "0.1264141065389438"
    25) test-error
    26) "0.13803731074639586"
    27) classification-errors-perc
    28) 19.00
    29) overfitting-detected
    30) yes

The network was trained for 0.29 seconds. At the end of the training,
that was stopped for overfitting, the error rate in the testing dataset
was 19%.

You can also specify to train for a given amonut of seconds or cycles.
For now we just use the `AUTOSTOP` feature since it is simpler. However we'll
dig into more details in the next section.

We can now show the output of the Ruby program after its full execution:

    47 prediction errors on 290 items

Does not look too bad, considering how simple is our model and the fact
we trained with just 401 data points. Modeling just on the percentage of
people that survived VS the ones that died, we could miss-predict more
than 100 passengers.

We can also play with a few variables interactively in order to check
what are the inputs that make a difference according to our trained
neural network.

Let's start asking the probable outcome for a woman, 30 years old,
first class, without siblings and parents:

    > NR.RUN mynet 1 0 0 0 1 30 0 0 200
    1) "0.093071778873849084"
    2) "0.90242156736283008"

The network is positive she survived, with 90% of probabilities.
What if she is a lot older than 30 years old, let's say 70?

    > NR.RUN mynet 1 0 0 0 1 70 0 0 200
    1) "0.11650946245068818"
    2) "0.88784839170875851"

This lowers her probability to 88.7%.
And if she traveled in third class with a very cheap ticket?

    > NR.RUN mynet 0 0 1 0 1 70 0 0 20
    1) "0.53693405013043127"
    2) "0.51547605838387811"

This time is 50% and 50%... Throw your coin.

The gist of this example is that, many problems you face as a developer
in order to optimize your application and do better choices in the
interaction with your users, are Titanic problems, but not in their
size, just in the fact that a simple model can "solve" them.

Overfitting detection and training tricks
===

One thing that makes neural networks hard to use in an interactive
way like the one they are proposed in this Redis module, is for sure
overfitting. If you train too much, the neural network ends to be
like that one student that can exactly tell you all the words in the
lesson, but if you ask a more generic question about the argument she
or he just wonders and can't reply.

So the `NR.TRAIN` command `AUTOSTOP` option attempts to detect
overfitting to stop the training before it's too late. How is this
performed? Well the current solution is pretty trivial: as the training
happens, we check the current error of the neural network between
the training dataset and the testing dataset.

When overfitting kicks in, usually what we see is that the network error
rate starts to be lower and lower in the training dataset, but instead
of also reducing in the testing dataset it inverts the tendency and
starts to grow. To detect this turning point is not simple for two
reasons:

1. The error may fluctuates as the network learns.
2. The network error may just go higher in the testing dataset since the learning is trapped into a *local minima*, but then a better solution may be found.

So while `AUTOSTOP` kinda does what it advertises (but I'll work on
improving it in the future, and there are neural network experts that
know much better than me and can submit a kind Pull Request :-), there
are also means to manually train the network, and see how the error
changes with training.

For instance, this is the error rate in the Titanic dataset after
the automatic stop:

    21) training-total-seconds
    22) 0.17
    23) dataset-error
    24) "0.13170509045457734"
    25) test-error
    26) "0.13433443241900492"
    27) classification-errors-perc
    28) 18.50

We can use the `MAXTIME` and `MAXCYCLES` options in order to train for
a specific amount of time (note that these options are also applicable
when `AUTOSTOP` is specified). Normally `MAXTIME` is set to 10000, which
are milliseconds, so to 10 seconds of total training before killing the
training thread. Let's train our network for 30 seconds, without auto stop.

    > NR.TRAIN mynet MAXTIME 30000
    Training has started

As a side note, while one or more trainings are in progress, we can
list them:

    > NR.THREADS
    1) nn_id=9 key=mynet db=0 maxtime=30000 maxcycles=0

After the training stops, let's show info again:

    21) training-total-seconds
    22) 30.17
    23) dataset-error
    24) "0.0674554189303056"
    25) test-error
    26) "0.20468644603795394"
    27) classification-errors-perc
    28) 21.50

You can see that our network overtrained: the error rate of the training
dataset is now lower: 0.06. But actually the performances in data it
never saw before, that is the testing dataset, is greater at 0.20!

And indeed, it classifies in a wrong way 21% of entries instead of 18.50%.

However it's not always like that, so to test things manually is a good
idea when working at machine learning experiments, especially with this
module that is experimental.

A more complex non linear classification example
===

The Titanic example is surely more interesting, however it is possible
that most relations between inputs and outputs are linear, so we'll now
try a non linear classification task, just for the sake of showing the
capabilities of a small neural network.

In the examples directory of this source distribution there is an example
called `circles.rb`, we'll use it as a reference.

We'll just setup a classification problem where the neural network
will be asked to classify two inputs, which are from our point of
view two coordinates in a 2D space, into three different classes:
0, 1 and 2.

While the neural network does not know this, we'll generate the data
so that different classes actually map to three different circles
in the 2D space: the circles also contain intersections. The function
that generates the dataset is the following:

```
    point_class = rand(3) # Class will be 0, 1 or 2
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
```

The basic trigonometric function:

    x = Math.sin(k)
    y = Math.cos(k)

With `k` going from 0 to 2*PI, is just a circle, so the above functions
are just circles, plus the `rand()` calls in order to introduce noise.
Basically if I trace the above three classes of points in a graphical
way with [load81](https://github.com/antirez/load81), I obtain the
following image:

![Circles plotted with LOAD81](http://antirez.com/misc/nn-circles.png)

The program `circles.rb`, it will generate the same set of points and
will feed them into the neural network configured to accept 2 inputs
and output one of three possible classes.

After about 2 seconds of training, we try to visualize what the neural
network has learned (also part of the `circles.rb` command) in this way:
for each point in an `80x80` grid, we ask the network to classify the
point. This is the ASCII-artist result:

```
................................................................................
................................................................................
................................................................................
................................................................................
................................................................................
................................................................................
................................................................................
................................................................................
/...............................................................................
///.............................................................................
////............................................................................
//////..........................................................................
///////.........................................................................
/////////.......................................................................
//////////......................................................................
////////////....................................................................
/////////////...................................................................
//////////////..................................................................
///////////////.................................................................
/////////////////...............................................................
//////////////////..............................................................
///////////////////.............................................................
////////////////////............................................................
/////////////////////...........................................................
//////////////////////..........................................................
////////////////////////........................................................
/////////////////////////.......................................................
//////////////////////////......................................................
///////////////////////////.....................................................
////////////////////....///////.................................................
///////////////////........//////...............................................
///////////////////.........///////.............................................
///////////////////..........////////...........................................
///////////////////...........///////...........................................
///////////////////...........///////...........................................
///////////////////............///////..........................................
///////////////////............///////..........................................
//////////////////.............///////..........................................
//////////////////.............///////..........................................
//////////////////............////////..........................................
//////////////////............////////..........................................
//////////////////............////////..........................................
///////////////////.........../////////.........................................
///////////////////..........//////////..OOOOOOOOO..............................
///////////////////..........//////////.OOOOOOOOOOOOOO..........................
///////////////////..........//////////OOOOOOOOOOOOOOOOOOOO.....................
///////////////////........../////////OOOOOOOOOOOOOOOOOOOOOOOOO.................
////////////////////......../////////OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO..
////////////////////........///////.OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
/////////////////////......///////.OOOOOOOOOOOOOOOOOOOOOOOOO..OOOOOOOOOOOOOOOOOO
/////////////////////......//////..OOOOOOOOOOOOOOOOOOOOOOO......OOOOOOOOOOOOOOOO
//////////////////////....//////...OOOOOOOOOOOOOOOOOOOOOO........OOOOOOOOOOOOOOO
//////////////////////////////......OOOOOOOOOOOOOOOOOOOO.........OOOOOOOOOOOOOOO
////////////////////////////........OOOOOOOOOOOOOOOOOOO.........OOOOOOOOOOOOOOOO
//////////////////////////...........OOOOOOOOOOOOOOOOO.........OOOOOOOOOOOOOOOOO
////////////////////////..............OOOOOOOOOOOOOO..........OOOOOOOOOOOOOOOOOO
///////////////////////....................OOOOOOOO..........OOOOOOOOOOOOOOOOOOO
//////////////////////.....................OOOOOOOO.........OOOOOOOOOOOOOOOOOOOO
/////////////////////......................OOOOOOO........OOOOOOOOOOOOOOOOOOOOOO
////////////////////........................OOOOO........OOOOOOOOOOOOOOOOOOOOOOO
///////////////////.........................OOOO.......OOOOOOOOOOOOOOOOOOOOOOOOO
//////////////////..........................OOOO.....OOOOOOOOOOOOOOOOOOOOOOOOOOO
/////////////////............................OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
////////////////.............................OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
///////////////........................../...OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
//////////////..........................//...OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
/////////////..........................////..OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
////////////.........................//////..OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
//////////..........................///////..OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
/////////..........................////////..OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
////////..........................//////////.OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
///////..........................///////////.OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
/////........................../////////////.OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
////..........................//////////////.OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
///..........................///////////////.OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
//........................../////////////////OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
...........................//////////////////OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
..........................///////////////////OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
.........................////////////////////OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
......................../////////////////////OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
```

As you can see, while the problem had no linear solution, the neural network
was able to split the 2D space into areas, with the *holes* where there
is the intersection between the circles, and thiner areas where there
are the intersections between the bigger circle and the smaller ones.

This example was not practical perhaps but shows well the power of the
neural network in non linear tasks.

API reference
===

In the above tutorial not all the options of all the commands may be
covered, so here there is a small reference with all the commands
supported by this extension and associated options.

### NR.CREATE key [CLASSIFIER|REGRESSOR] inputs [hidden-layer-units ...] -> outputs [NORMALIZE] [DATASET maxlen] [TEST maxlen]

Create a new neural network if the target key is empty, or returns an error.

* key - The key name holding the neural network.
* CLASSIFIER or REGRESSOR is the network type, read this tutorial for more info.
* inputs - Number of input units
* hidden-layer-units zero or more arguments indicating the number of hidden units, one number for each layer.
* outputs - Number of outputs units
* NORMALIZE - Specify if you want the network to normalize your inputs. Use this if you don't know what we are talking about.
* DATASET maxlen - Max number of data samples in the training dataset.
* TEST maxlen - Max number of data samples in the testing dataset.

Example:

    NR.CREATE mynet CLASSIFIER 64 100 -> 10 NORMALIZE DATASET 1000 TEST 500

### NR.OBSERVE key i0 i1 i2 i3 i4 ... iN -> o0 o1 o3 ... oN [TRAIN|TEST]

Add a data sample into the training or testing dataset (if specified as last argument) or evenly into one or the other, according to their respective sizes, if no target is specified.

For neural networks of type CLASSIFIER the output must be just one, in the range from 0 to `number-of-outputs - 1`. It's up to the network to translate the class ID into a set of zeros and ones.

The command returns the number of data samples inside the training and testing dataset. If the target datasets are already full, a random entry is evicted and substituted with the new data.

## NR.RUN key i0 i1 i2 i3 i4 ... iN

Run the network stored at key, returning an array of outputs.

## NR.CLASS key i0 i1 i2 i3 i4 ... iN

Like `NR.RUN` but can be used only with NNs of type CLASSIFIER. Instead of outputting the raw neural network outputs, the command returns the output class directly, which is, the index of the output with the greatest value.

## NR.TRAIN key [MAXCYCLES count] [MAXTIME milliseconds] [AUTOSTOP]

Train a network in a background thread. When the training finishes
automatically updates the weights of the trained networks with the
new ones and updates the training statistics.

The command works with a copy of the network, so it is possible to
use the network while it is undergoing a training.

If no AUTOSTOP is specified, trains the network till the maximum number of
cycles or milliseconds are reached. If no maximum number of cycles is specified
there are no cycles limits. If no milliseconds are specified, the limit is
set to 10000 milliseconds (10 seconds).

If AUTOSTOP is specified, the training will still stop when the maximum 
umber of cycles or milliseconds is specified, but will also try to stop
the training if overfitting is detected. Check the previous sections for
a description of the (still naive) algorithm the implementation uses in
order to stop.

## NR.INFO key

Show many internal information about the neural network. Just try it :-)

## NR.THREADS

Show all the active training threads.

Contributing
===

The main aim of Neural Redis, which is currently just a 48h personal
hackatlon, is to show the potential that there is in an accessible API
that provides a simple to use machine learning tool, that can be used
and trained interactively.

However the neural network implementation can be surely improved in different
ways, so if you are an expert in this field, feel free to submit changes
or ideas. One thing that I want to retain is the simplicity of the outer
layer: the API. However the techniques used in the internals can be more
complex in order to improve the results.

There is to note that, given the API exported, the implementation of
the neural network should be, more than state of art in solving a specific
problem, more designed in order to work well enough in a large set of
conditions. While the current fully connected network has its limits,
it together with BPROP learning shows to be quite resistant to misuses.
So an improved version should be able to retain, and extend this quality.
The simplest way to guarantee this is to have a set of benchmarks of different
types using open datasets, and to score different implementations against
it.

Plans
===

* Better overfitting detection.
* Implement RNNs with a simpler to use API.
* Use a different loss function for classification NNs.
* Get some ML expert which is sensible to simple APIs involved.

Have fun with machine learning,

Salvatore
