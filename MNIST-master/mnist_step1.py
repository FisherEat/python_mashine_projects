# 本案例是 极客学院MNIST手写识别:(步骤一, 简单入门)
# http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Get MNIST Data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Variables
batch_size = 100
total_steps = 5000
steps_per_test = 100

# Build Model
x = tf.placeholder(tf.float32, [None, 784])
y_label = tf.placeholder(tf.float32, [None, 10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, w) + b)

# Loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Prediction
correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_label, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Train 10000 steps
    for step in range(total_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={x: batch_x, y_label: batch_y})
        # Test every 100 steps
        if step % steps_per_test == 0:
            print(step, sess.run(accuracy, feed_dict={x: mnist.test.images, y_label: mnist.test.labels}))

# y 为预测结果
# y_label为实际结果,已经标注好的结果
# x 为训练集 , 因此这里的训练集用的是(x, y_label), 计算预测精度和准确度用的是预测集(x, y)
# 这个案例和tf_test_bp_all 类似, 在x相同的情况下,比较 预测集y和实际结果y_label之间的偏差.
