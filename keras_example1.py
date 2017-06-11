from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# to make tests deterministic its necessary to set fixed random seed
np.random.seed(7)

# load pima indians dataset
dataset = np.loadtxt("Data/pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# creating model
input_size = X.shape[1]

model = Sequential()
model.add(Dense(12, input_dim=input_size, activation='relu'))
model.add(Dense(input_size, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X, Y, epochs=150, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)

#round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
