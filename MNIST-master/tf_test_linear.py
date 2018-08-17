# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# 本案例用来测试线性规划

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def tf_linear_plt():
    # 模拟生成100对数据对, 对应的函数为y = x * 0.1 + 0.3
    x_data = np.random.rand(100).astype("float32")
    y_data = x_data * 0.1 + 0.3

    # 指定w和b变量的取值范围（利用TensorFlow来得到w和b的值）
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) #随机生成一个在[-1,1]范围的均匀分布数值
    b = tf.Variable(tf.zeros([1])) #set b=0
    y = W * x_data + b

    # 最小化均方误差
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5) #学习率为0.5的梯度下降法
    train = optimizer.minimize(loss)

    # 初始化TensorFlow参数
    init = tf.initialize_all_variables()

    # 运行数据流图（
    sess = tf.Session()
    sess.run(init)

    # 观察多次迭代计算时，w和b的拟合值
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(W), sess.run(b))


def tf_linear_normal():
    num_points = 1000
    vectors_set = []
    for i in range(num_points):
        x1 = np.random.normal(0.0, 0.55)
        y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
        vectors_set.append([x1, y1])

    x_data = [v[0] for v in vectors_set]
    y_data = [v[1] for v in vectors_set]

    # Graphic display
    plt.plot(x_data, y_data, 'ro')
    plt.legend()
    plt.show()

    W = tf.Variable(tf.random_uniform([1], -1, 1))
    b = tf.Variable(tf.zeros[1])
    y = W * x_data + b

    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.05)
    train = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for step in range(101):
            print(step, sess.run(W), sess.run(b))
            print(step, sess.run(loss))

            #Graphic display
            plt.plot(x_data, y_data, 'ro')
            plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
            plt.xlabel('x')
            plt.xlim(-2,2)
            plt.ylim(0.1,0.6)
            plt.ylabel('y')
            plt.legend()


if __name__ == '__main__':
    tf_linear_normal()