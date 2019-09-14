import tensorflow as tf
import numpy as np
import sys

# set the random seeds to make sure your results are reproducible
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

# specify path to training data and testing data, using the parameters entered in the command line
train_x_location = sys.argv[1]
train_y_location = sys.argv[2]
test_x_location = sys.argv[3]
test_y_location = sys.argv[4]

print("Reading training data")
x_train = np.loadtxt(train_x_location, dtype="uint8", delimiter=",")
y_train = np.loadtxt(train_y_location, dtype="uint8", delimiter=",")

m, n = x_train.shape # m training examples, each with n features
m_labels,  = y_train.shape # m2 examples, each with k labels
l_min = y_train.min()

assert m_labels == m, "x_train and y_train should have same length."
assert l_min == 0, "each label should be in the range 0 - k-1."
k = y_train.max()+1

print(m, "examples,", n, "features,", k, "categories.")

#preprocess the data, divide the data set values by 100
print("Pre processing x of training data")
x_train = x_train / 100

# define the training model
model = tf.keras.models.Sequential([
    # input_shape should be specified in the first layer
    tf.keras.layers.Dense(25, activation=tf.keras.activations.relu,
                          input_shape=(n,)),
    # another layer
    tf.keras.layers.Dense(25, activation=tf.keras.activations.relu),
    # another layer with l2 regularization
    tf.keras.layers.Dense(25, activation=tf.nn.relu,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0095, l2=0.0095)),
    # dropouts layer
    tf.keras.layers.Dropout(0.075),
    # last layer is softmax
    tf.keras.layers.Dense(k, activation=tf.nn.softmax)
])
# loss='categorical_entropy' expects input to be one-hot encoded
# loss='sparse_categorical_entropy' expects input to be the category as a number
# options for optimizer: 'sgd' and 'adam'. sgd is stochastic gradient descent
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("train")
model.fit(x_train, y_train, epochs=1000, batch_size=32
)
# default batch size is 32


print("Reading testing data")
x_test = np.loadtxt(test_x_location, dtype="uint8", delimiter=",")
y_test = np.loadtxt(test_y_location, dtype="uint8", delimiter=",")

m_test, n_test = x_test.shape
m_test_labels,  = y_test.shape
l_min = y_train.min()

assert m_test_labels == m_test, "x_test and y_test should have same length."
assert n_test == n, "train and x_test should have same number of features."

print(m_test, "test examples.")

print("Pre processing testing data")
x_test = x_test / 100

print("evaluate")
model.evaluate(x_test, y_test)
