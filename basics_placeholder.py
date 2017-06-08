import tensorflow as tf

a = tf.placeholder(tf.float32, shape=[3, 4])
b = tf.Variable(tf.truncated_normal([4, 3]))
c = tf.matmul(a, b)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(init)
    print(sess.run(c, feed_dict={a: [[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6]]}))
