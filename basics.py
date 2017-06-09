import tensorflow as tf

#a = tf.constant(3, name='a')
a = tf.Variable(2, name='a_variable')
b = tf.Variable(2, name='b')
x = tf.multiply(a, b, name='x')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./graphs", sess.graph)

    print sess.run(init)
    print sess.run(x)
    # print sess.run(x.value())

writer.close()