import tensorflow as tf
import numpy as np

# 随机函数
# norm = tf.random_normal([2, 3], mean=-1, stddev=4)
norm = tf.random_normal([2, 3], mean=-1, stddev=4)
c = tf.constant([[1, 2], [3, 4], [5, 6]])
shuff = tf.random_shuffle(c)

d = tf.truncated_normal([2, 2], mean=3, stddev=1.0)
f = tf.random_uniform([3, 4], minval=0, maxval=None)
g = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
# sess = tf.Session()
# print(sess.run(norm))
# print(sess.run(norm))

# shape相关方法
x = tf.constant([[1, 2, 3], [4, 5, 6]])
y = [[1, 2, 3], [4, 5, 6]]
z = np.arange(24).reshape([2, 3, 4])

sess = tf.Session()
print(sess.run(tf.shape(x)))
print(sess.run(tf.shape(y)))
print(sess.run(tf.size(x)))

test_reduce = tf.constant([[1, 1, 1], [1, 1, 1]])
print("hhha :", sess.run(tf.reduce_sum(test_reduce)))
print("hhhh :", sess.run(tf.reduce_sum(test_reduce, 0)))

# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     sess.run(norm)
#     sess.run(shuff)
#     sess.run(d)
#     sess.run(f)
#     # print(sess.run(norm))
#     # print(sess.run(shuff))
#     # print(sess.run(d))
#     # print(sess.run(f))
#     # print(tf.shape(g))
