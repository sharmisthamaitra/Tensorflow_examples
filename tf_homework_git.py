import tensorflow as tf
import numpy as np
import random as rnd


with tf.name_scope("Input_placeholder"):
    a = tf.placeholder(tf.float32, shape=None, name='a_placeholder')
    
    mean=1
    std=2
    random_nbrs=np.random.normal(mean, std, 100)


with tf.name_scope("Middle_Section"):
    b = tf.reduce_prod(a, name="b_prod_a")

    c = tf.reduce_mean(a, name="c_mean_a")

    d = tf.reduce_sum(a, name="d_sum_a")

    e = tf.add(c, b, name="e_sum_b_c")


with tf.name_scope("Final_Node"):
    f = tf.multiply(e,d, name="f_mul_d_e")

print(a)
print(b)
print(c)
print(d)
print(e)
print(f)


sess = tf.Session()
tf.reset_default_graph() 

sess.run(a, feed_dict={a: random_nbrs})
sess.run(b, feed_dict={a: random_nbrs})
sess.run(c, feed_dict={a: random_nbrs})
sess.run(d, feed_dict={a: random_nbrs})
sess.run(e, feed_dict={a: random_nbrs})
sess.run(f, feed_dict={a: random_nbrs})

writer = tf.summary.FileWriter('./tf_homework', sess.graph)
writer.close()






