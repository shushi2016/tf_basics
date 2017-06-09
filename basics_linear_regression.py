import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import xlrd

DATA_FILE = 'fire_theft.xls'

# Phase 1: Assemble the graph
# Step 1: read in data file
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.array([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
X = tf.placeholder(dtype=tf.float32, name='X')
Y = tf.placeholder(dtype=tf.float32, name='Y')

# Step 3: create weights and bias, initialized to 0
# name your variables w and b
w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

# Step 4: predict Y (number of theft) from the number of fire
# name your variabel Y_predicted
Y_predicted = X * w + b

# Step 5: use the square error as the loss function
# name your variable loss
loss = tf.square(Y - Y_predicted, name='loss')

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# Phase 2 Train the model
with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/03/linear_reg', sess.graph)

    # Step 8: train the model
    for i in range(300):  # run 100 epochs
        total_loss = 0
        for x, y in data:
            # Session runs optimizer to minimize loss and fetch the value of loss
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += l
        plt.plot(i, total_loss, 'ro', label='total_loss')
        print("Epoch {0}: {1}".format(i, total_loss / n_samples))
    plt.show()

    writer.close()
#
# # plot the results
# X, Y = data.T[0], data.T[1]
# Yhat = X * w + b
# plt.plot(X, Y, 'bo', label='Real data')
# # plt.plot(X, X * w + b, 'r', label='Predicted data')
# #plt.plot(X, Yhat, 'r', label='Predicted data')
# plt.legend()
# plt.show()
