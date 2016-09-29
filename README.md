Neural Redis is a Redis loadable module that implements feed forward neural
networks as a native data type for Redis. The project goal is to provide
Redis users with an extremely simple to use machine learning experience.

Normally machine learning is operated by collecting data, traning some
system, and finally executing the resulting program in order to solve actual
problems. In Neural Redis all this phases are compressed into a single API:
the data collectin and training all happen inside the Redis server.
Neural networks can be executed while there is an ongoing training, and can
be re-trained multiple times as new data from the outside is collected
(for instance user events).

The project starts from the observation that, while complex problemss like
computer vision need slow to train and complex neural networks setups, many
regression and classification problems that are able to enhance the user
experience in many applications, are approachable by feed forward fully
connected small networks, that are very fast to rain, very generic, and
robust to far from perfect parameters configurations.

Neural Redis implements:

* A very simple to use API.
* Automatic data normalization.
* Online training of neural networks in different threads.
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
more than 1000 lines of C code composing this extension in roughly
two days.

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
is used in order to detect is the network is able to generalize, that is,
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

AS you can see we have 6 dataset items and 2 test items. We configured
the network at creation time to have space for 50 and 10 items. As you add
items with `NR.OBSERVE` the network will put items evenly on both datasets,
proportionally to their respective size. Finally when the datasets are full,
old random entries are replaced with new
ones.

We can also see that the network was trained with 1344 step for 0 seconds
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
inputs rappresenting *something*, to label it with one of a fixed set of
labels.

For example the inputs may be features of Greek jars, and the classification
output could be one of the following three jar types:

* Type 0: Kylix type A
* Type 1: Kylix type B
* Type 2: Kassel cup

As a programmer you may think that, the output class, is just a single output
number. However neural networks don't work well this way, for example
classifying type 0 with an output between 0 and 0.33, type 1 with an output
between 0.33 and 0.66, and finally type 2 with an output between 0.66 and 1.

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
greater than the others. THere is a Neural Redis command that saves you the
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

Listing training threads
===

Overfitting detection and training tricks
===


