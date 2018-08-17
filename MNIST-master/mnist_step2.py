# 本案例是 极客学院MNIST手写识别:(步骤二, 深入了解MNIST)
# http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_pros.html

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Variables
batch_size = 100
total_steps = 5000
dropout_keep_prob = 0.5
steps_per_test = 100


def weight(shape, stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)


def bias(shape, value=0.1):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)


def conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(input, filter, strides=strides, padding=padding)


def max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding)

# Initial
x = tf.placeholder(tf.float32, shape=[None, 784])
y_label = tf.placeholder(tf.float32, shape=[None, 10])
x_reshape = tf.reshape(x, [-1, 28, 28, 1])

# Layer1
w_conv1 = weight([5, 5, 1, 32])
b_conv1 = bias([32])
h_conv1 = tf.nn.relu(conv2d(x_reshape, w_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)

# Layer2
w_conv2 = weight([5, 5, 32, 64])
b_conv2 = bias([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2)

# Layer3
w_fc1 = weight([7 * 7 * 64, 1024])
b_fc1 = bias([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

# Softmax
w_fc2 = weight([1024, 10])
b_fc1 = bias([10])
y = tf.nn.softmax(tf.matmul(h_fc1_dropout, w_fc2) + b_fc1)

# Loss
cross_entropy = -tf.reduce_sum(y_label * tf.log(y))
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Prediction
correct_prediction = tf.equal(tf.argmax(y_label, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(total_steps + 1):
        batch = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={x: batch[0], y_label: batch[1], keep_prob: dropout_keep_prob})
        # Train accuracy
        if step % steps_per_test == 0:
            print('Training Accuracy', step,
                  sess.run(accuracy, feed_dict={x: batch[0], y_label: batch[1], keep_prob: 1}))

# Final Test
print('Test Accuracy', sess.run(accuracy, feed_dict={x: mnist.test.images, y_label: mnist.test.labels,
                                                     keep_prob: 1}))
# y 为预测结果
# y_label为实际结果,已经标注好的结果
# x 为训练集 , 因此这里的训练集用的是(x, y_label), 计算预测精度和准确度用的是预测集(x, y)
# 这个案例和tf_test_bp_all 类似, 在x相同的情况下,比较 预测集y和实际结果y_label之间的偏差.
# 这个案例与mnist_step1 不同之处在于求解y的时候, 使用了卷积网络cnn
