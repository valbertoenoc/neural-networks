import numpy as np
import matplotlib.pyplot as plt

# Basic Neural Network to solve XOR

#sigmoid function put output in [0,1] range
def sigmoid(z, deriv=False):
	if deriv:
		return z*(1-z)
	return 1/(1 + np.exp(-z))

# input data
# (last column represents bias)
X = np.array([[0,0,1],
			  [0,1,1],
			  [1,0,1],
			  [1,1,1]])

# desired output
y = np.array([[0,0,1,1]]).T

# Initializing weights with mean 0
w0 = 2*np.random.random((X.shape[1], X.shape[0])) - 1
w1 = 2*np.random.random((X.shape[0], 1)) - 1

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# training phase
n_epochs = 60000
acc_error = []
for i in xrange(n_epochs):
	l0 = X 
	l1 = sigmoid(np.dot(l0, w0)) 
	l2 = sigmoid(np.dot(l1, w1))

	l2_err = y - l2
	l2_delta = l2_err + sigmoid(l2, True)

	acc_error.append(np.mean(np.abs(l2_err)))
	if (i % 10000) == 0:
		print("Error: " + str(np.mean(np.abs(l2_err))))

	l1_err = np.dot(l2_delta, w1.T)
	#l1_err = l2_delta.dot(w1)
	l1_delta = l1_err + sigmoid(l1, True)


	w1 += np.dot(l1.T, l2_delta)
	w0 += np.dot(l0.T, l1_delta)

# final model output
print(l2)
print('-'*40)

#sample input to test model prediction
input_test = np.array([[1,1,1]])
l1 = sigmoid(np.dot(input_test, w0))
l2 = sigmoid(np.dot(l1, w1))
print(l2)

#plotting error curve
plt.plot(range(n_epochs), acc_error)
plt.show()
