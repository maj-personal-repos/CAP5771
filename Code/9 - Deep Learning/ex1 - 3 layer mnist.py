import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Python optimisation variables
learning_rate = 0.5
epochs = 10
batch_size = 100

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784], name="x")

# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10], name="labels")

# add image to summary
x_image = tf.reshape(x, [-1, 28, 28, 1])
tf.summary.image("input", x_image, 10)

# define first layer - relu activated hidden layer
with tf.name_scope("hidden_layer"):
    W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random_normal([300]), name='b1')
    hidden_out = tf.add(tf.matmul(x, W1), b1)
    hidden_out = tf.nn.relu(hidden_out)

# define the output layer - softmax activated output layer
with tf.name_scope("output_layer"):
    W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random_normal([10]), name='b2')
    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2), name='y_')

with tf.name_scope('xent'):
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))
    tf.summary.scalar('xent', cross_entropy)

# add an optimiser
with tf.name_scope('train'):
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# define an accuracy assessment operation
with tf.name_scope('train'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# Initialize
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('/tmp/3_layer_mnist/demo/1')
writer.add_graph(sess.graph)

total_batch = int(len(mnist.train.labels) / batch_size)

for epoch in range(epochs):
    avg_cost = 0
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)

        if i % 5 == 0:
            [train_accuracy, s] = sess.run([accuracy, merged_summary], feed_dict={x: batch_x, y: batch_y})
            writer.add_summary(s, epoch*total_batch + i)
            # print("Epoc:", (epoch + 1), "Step:", (i), "training accuracy %g" % train_accuracy)

        [_, c] = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
        avg_cost += c / total_batch

    print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))