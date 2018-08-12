# Most of part 1 of the code is taken directly from 
# https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/ 

# In Part 2 I try to generate visualizations of saliency maps for the
# digit images by using backpropagation and guided backpropagation 

#Part1

# Imports
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imresize
from scipy.io import savemat
from scipy.io import loadmat

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],\
        padding='SAME')


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Input Layer
x_image = tf.reshape(x, [-1,28,28,1])

# Convolutional Layer #1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Convolutional Layer #2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# FC Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout Regularization
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# FC Layer #2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Softmax
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits\
    (labels=y_, logits=y_conv))

# Training and testing
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1],\
                keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
    W_conv1_data = sess.run(W_conv1)
    b_conv1_data = sess.run(b_conv1)
    W_conv2_data = sess.run(W_conv2)
    b_conv2_data = sess.run(b_conv2)
    W_fc1_data = sess.run(W_fc1)
    b_fc1_data = sess.run(b_fc1)
    W_fc2_data = sess.run(W_fc2)
    b_fc2_data = sess.run(b_fc2)
    print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, \
        y_: mnist.test.labels, keep_prob: 1.0}))

savemat('cnn_data', {'W_conv1':W_conv1_data, 'b_conv1':b_conv1_data, \
    'W_conv2':W_conv2_data, 'b_conv2':b_conv2_data, 'W_fc1': W_fc1_data, \
    'b_fc1': b_fc1_data, 'W_fc2': W_fc2_data, 'b_fc2':b_fc2_data})
    
cnn_data = loadmat('cnn_data.mat')



#Part2


# Backpropagation
W_conv1 = tf.constant(W_conv1_data)
b_conv1 = tf.constant(b_conv1_data)
W_conv2 = tf.constant(W_conv2_data) 
b_conv2 = tf.constant(b_conv2_data)
W_fc1 = tf.constant(W_fc1_data)
b_fc1 = tf.constant(b_fc1_data)
W_fc2 = tf.constant(W_fc2_data)
b_fc2 = tf.constant(b_fc2_data) 
    
# Redefine graph
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Input Layer
x_image = tf.reshape(x, [-1,28,28,1])

# Convolutional Layer #1
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Convolutional Layer #2
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# FC Layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout Regularization
keep_prob = 1.0
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# FC Layer #2
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    

test_image = np.array([mnist.test.images[17]])
test_class = np.array([mnist.test.labels[17]])
onehot = tf.reshape(tf.constant(test_class, dtype=tf.float32), shape=[10,1])
probs = tf.reshape(tf.nn.softmax(y_conv), [1,10])
probability = tf.matmul(probs,onehot)
gradient = tf.gradients(probability, x)

    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saliency = sess.run(gradient, feed_dict={x:test_image})[0].reshape(28,28)

saliency_map = (saliency > 0)*saliency 
plt.imshow(saliency_map, cmap = 'binary_r')
plt.show()
    

# Guided backpropagation

grad_fc2 = tf.nn.relu(tf.gradients(probability,y_conv)[0])
grad_fc1 = tf.nn.relu(tf.gradients(y_conv, h_fc1, grad_ys=grad_fc2)[0])
grad_pool2 = tf.nn.relu(tf.gradients(h_fc1, h_pool2, grad_ys=grad_fc1)[0])
grad_conv2 = tf.nn.relu(tf.gradients(h_pool2, h_conv2, grad_ys=grad_pool2)[0])
grad_pool1 = tf.nn.relu(tf.gradients(h_conv2, h_pool1, grad_ys=grad_conv2)[0])
grad_conv1 = tf.nn.relu(tf.gradients(h_pool1, h_conv1, grad_ys=grad_pool1)[0])
grad_x = tf.nn.relu(tf.gradients(h_conv1, x, grad_ys=grad_conv1)[0])

# Example of comparison between Guided-BP visualization and real image
with tf.Session() as sess:
    saliency2 = sess.run(grad_x, feed_dict={x:test_image})[0].reshape(28,28)
    
saliency_map2 = saliency2 
plt.figure(2)
plt.imshow(saliency_map2, cmap = 'binary_r')
plt.show()

plt.figure(3)
plt.imshow(mnist.test.images[17].reshape(28,28), cmap = 'binary_r')
plt.show()


   

