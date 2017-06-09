import tensorflow as tf

my_var=tf.Variable(2.0)

my_var_times_two = my_var.assign(2*my_var)

with tf.Session() as sess:
	print sess.run(my_var.initializer)
	print sess.run(my_var_times_two)
	print sess.run(my_var_times_two)
	print sess.run(my_var_times_two)