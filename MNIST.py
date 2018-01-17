import tensorflow as tf
import numpy as np 
import math
import scipy 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def create_placeholders(n_x, n_y):
	X = tf.placeholder(tf.float32, [None, n_x], name = 'X')
	Y = tf.placeholder(tf.float32, [None, n_y], name = 'Y')
	return X, Y

def initialize_parameters():
	W1 = tf.get_variable("W1", [784, 10], initializer = tf.contrib.layers.xavier_initializer())
	b1 = tf.get_variable("b1", [10], initializer = tf.zeros_initializer())
	parameters = {'W1' : W1, 'b1' : b1}
	return parameters

def forward_propagation(X, parameters):
	W1 = parameters['W1']
	b1 = parameters['b1']
	Z = tf.nn.softmax(tf.add(tf.matmul(X, W1) , b1))
	return Z 

def compute_cost(Z3, Y):
	logits = tf.transpose(Z3)
	labels = tf.transpose(Y)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
	return cost

def random_mini_batches(X, Y, mini_batch_size):
	m = X.shape[0]
	#print('m = ' + str(m))
	mini_batches = []

	#shuffling
	permutation = list(np.random.permutation(m))
	shuffled_X = X[permutation, :]
	shuffled_Y = Y[permutation, :].reshape(m, 10)
	#print('shuffled_X = ' + str(shuffled_X.shape))
	#print('shuffled_Y = ' + str(shuffled_Y.shape))
	#partitioning
	num_complete_minibatches = math.floor(m/mini_batch_size)

	for k in range (0, num_complete_minibatches):
		mini_batch_X = shuffled_X[k * mini_batch_size:(k + 1)* mini_batch_size, : ]
		mini_batch_Y = shuffled_Y[k * mini_batch_size:(k + 1)* mini_batch_size, : ]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	#with this dataset, we have no end case. 

	return mini_batches
	
def model (X_train, Y_train, X_test, Y_test, learning_rate = 0.05, num_epochs = 200, minibatch_size = 20, print_cost = True):
	print ('Xtraindims = ' + str(X_train.shape))
	print ('Ytraindims = ' + str(Y_train.shape))
	(m, n_x) = X_train.shape 
	n_y = Y_train.shape[1]
	costs = []
	X, Y = create_placeholders(n_x, n_y)
	print('placeholders created =' + str(X.shape) + str(Y.shape))

	parameters = initialize_parameters()	
	print('parameters initialized')

	Z = forward_propagation(X, parameters)
	print('fwd prop done')

	cost = compute_cost(Z, Y)
	print('cost computed')

	optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
	print('optimizer defined')
	print(Y_train.shape)

	init = tf.global_variables_initializer()
	
	with tf.Session() as sess:
		sess.run(init)	
		print('session started')

		for epoch in range(num_epochs):
			print ('epoch ' + str(epoch))
			epoch_cost = 0.                       # Defines a cost related to an epoch
			num_minibatches = int(int(m) / minibatch_size) # number of minibatches of size minibatch_size in the train set
			minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
       
			for minibatch in minibatches:
				# Select a minibatch
				(minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
				#print ('mninbX = ' + str(minibatch_X.shape))
				#print ('mninbY = ' + str(minibatch_Y.shape))
				_ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###
				#print(type(minibatch_cost))
				#print(type(num_minibatches))
				epoch_cost += minibatch_cost / num_minibatches

            	# Print the cost every epoch
			if print_cost == True and epoch % 1 == 0:
				print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
			if print_cost == True and epoch % 5 == 0:
				costs.append(epoch_cost)
                
     
        # lets save the parameters in a variable
		parameters = sess.run(parameters)
		print("Parameters have been trained!")

        # Calculate the correct predictions
		correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))

        # Calculate accuracy on the test set
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
		print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
		return parameters

#Now run the model

parameters = model(mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels)
