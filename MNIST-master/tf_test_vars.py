
# 本案例用来测试TensorFlow常用的常量和变量计算

from __future__ import print_function, division
import tensorflow as tf

x = tf.placeholder(tf.float32, [1024, 1024])
y = tf.matmul(x, x)
print(x)

a1 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a1')
a2 = tf.Variable(tf.constant(1), name='a2')
a3 = tf.Variable(tf.ones(shape=[2, 3]), name='a3')
a4 = tf.Variable(tf.zeros([1, 1]))

# 例子二
state = tf.Variable(0, name='state')
print("the name of this variable:", state.name)

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# feed example
print("feed example")
input1 = tf.placeholder(tf.float32, [1])
input2 = tf.placeholder(tf.float32, [1])
output = tf.multiply(input1, input2)

# Graph和Session
print("Graph and Session example.")
c = tf.constant(value=1)
print(c.graph)
print(tf.get_default_graph())

g = tf.Graph()
print("g: ", g)
with g.as_default():
    d = tf.constant(value=2)
    print(d.graph)

with g.device('/gpu:0'):
    with g.device(None):
        print(g.device)

print("all keys :", g.get_all_collection_keys())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a1))
    print(sess.run(a2))
    print(sess.run(a3))
    print(sess.run(a4))
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print("value of state:", sess.run(state))
    result_feed = sess.run(output, feed_dict={input1: [2.], input2: [3.]})
    print(result_feed)
