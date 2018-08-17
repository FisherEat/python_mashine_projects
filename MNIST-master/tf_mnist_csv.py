
# 本案例将MNIST数据转化成csv格式,然后读取再做回归
# 验证一个猜想: 本质上神经网络是在做回归运算.
# 第一步: 将MNIST数据转换成 矩阵形式
# 第二步: 根据矩阵形式,建立回归模型,一般都是线性回归模型 y = w*x + b
# 第三步: 设置损失函数
# 第四步: 训练集训练
# 第五步: 测试集计算精度
# 第六步: 验证集验证回归

import tensorflow as tf
import pandas as pd

# Variables
batch_size = 100
total_steps = 5000
steps_per_test = 100

train_labels = pd.read_csv('./train_labels.csv')
train_data = pd.read_csv('./train_img.csv')

test_labels = pd.read_csv('./test_labels.csv')
test_data = pd.read_csv('./test_img.csv')

x = tf.placeholder(tf.float32, [None, 784])
y_label = tf.placeholder(tf.float32, [None, 1])
w = tf.Variable(tf.zeros([784, 1]))
b = tf.Variable(tf.zeros([1]))
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
        sess.run(train, feed_dict={x: train_data, y_label: train_labels})
        # Test every 100 steps
        if step % steps_per_test == 0:
            print(step, sess.run(accuracy, feed_dict={x: test_data, y_label: test_labels}))


