import tensorflow as tf

LOGDIR = "/tmp/mnist_tutorial/"
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + "data", one_hot=True)


# Define a simple convolutional layer
def conv_layer(input, channels_in, channels_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.zeros([5, 5, channels_in, channels_out]))
        b = tf.Variable(tf.zeros([channels_out]))
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# Define a fully connected layer
def fc_layer(input, channels_in, channels_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.zeros([channels_in, channels_out]))
        b = tf.Variable(tf.zeros([channels_out]))
        act = tf.nn.relu(tf.matmul(input, w) + b)
        return tf.nn.relu(tf.matmul(input, w) + b)


# Setup placeholders, and reshape the data
# mnist images are 28 x 28 = 784
# output should be 10 to correspond to 10 digits
# need to turn mnist vectors into an image
x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
x_image = tf.reshape(x, [-1, 28, 28, 1])

# create the network
conv1 = conv_layer(x_image, 1, 32, "conv1")
conv2 = conv_layer(conv1, 32, 64, "conv2")

flattened = tf.reshape(conv2, [-1, 7 * 7 * 64])
fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
logits = fc_layer(fc1, 1024, 10, "fc2")

# Compute cross entropy as the loss function
with tf.name_scope("xent"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# Use an AdamOptimizer to train the network
with tf.name_scope("train"):
    train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)

# Compute the accuracy
with tf.name_scope("accuracy"):
    correct_predition = tf.equal(tf.arg_max(logits, 1), tf.arg_max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))

# Initialize the variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train for 2000 steps
for i in range(2000):
    batch = mnist.train.next_batch(100)

    # Occasionally report accuracy
    if i % 500 == 0:
        [train_accuracy] = sess.run([accuracy], feed_dict={x: batch[0], y: batch[1]})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    # Run the training step
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

writer = tf.summary.FileWriter("/tmp/mnist_demo/1")
writer.add_graph(sess.graph)
