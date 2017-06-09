import tensorflow as tf

# Use a placeholder to create an 3*4 matrix
# There is nothing in a as of now.
a = tf.placeholder(tf.float32, shape=[3, 4])

# b is a 4*3 matrix, which need to be initialized later
b = tf.Variable(tf.truncated_normal([4, 3]))

# c is an matrix multiply op
c = tf.matmul(a, b)

# Initializer
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # write graph so later it can be viewed in tensorboard
    # command line should be:
    # tensorboard --logdir="./graphs" --port 60006
    writer = tf.summary.FileWriter('./graphs', sess.graph)

    # Now the variables are been initialized
    sess.run(init)

    # Use a feed_dict to put values to a, so the c can be computed
    print(sess.run(c, feed_dict={a: [[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6]]}))
