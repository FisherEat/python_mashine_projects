
'''

本案例是tensorflow 回归模型,采用tensorflow自带的梯度下降法生成回归曲线

'''

# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Prepare train data
# train_X = np.linspace(-1, 1, 100)
# train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10
#
# # Define the model
# X = tf.placeholder("float")
# Y = tf.placeholder("float")
# w = tf.Variable(0.0, name="weight")
# b = tf.Variable(0.0, name="bias")
# loss = tf.square(Y - X*w - b)
# train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
#
# # Create session to run
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     epoch = 1
#     for i in range(10):
#         for(x, y) in zip(train_X, train_Y):
#             _, w_value, b_value = sess.run([train_op, w, b], feed_dict={X: x, Y: y})
#             print("Epoch: {}, w: {}, b: {}".format(epoch, w_value, b_value))
#             epoch += 1
#
#
# # draw
# plt.plot(train_X, train_Y, "+")
# plt.plot(train_X, train_Y.dot(w_value) + b_value)
# plt.show()

'''

本案例是tensorflow 回归模型,采用tensorflow自带的梯度下降法生成回归曲线

'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Prepare train data
train_X = np.linspace(-1, 1, 100)
    # np.array([0.23, 0.34, 0.67, 0.89, 0.90, 0.97])
    # np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10
    # np.array([0.25, 0.33, 0.65, 0.80, 0.89, 0.99])
    # 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10

# Define the model
X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")
loss = tf.square(Y - X*w - b)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# Create session to run
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    epoch = 1
    for i in range(100):
        for (x, y) in zip(train_X, train_Y):
            _, w_value, b_value = sess.run([train_op, w, b], feed_dict={X: x, Y: y})
        print("Epoch: op: {}, {}, w: {}, b: {}".format(epoch, _, w_value, b_value))
        epoch += 1


# draw
plt.plot(train_X, train_Y, "+")
plt.plot(train_X, train_X.dot(w_value)+b_value)
plt.show()