import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split

# An Artificial Neural Network (ANN) consists of units (nodes) that are connected by weighted edges.
# The value of a unit is calculated by multiplying the input values (from the input units) by the weights,
# sum them up and adding a bias. The result is passed to an activation function g.
# output = g(sum_over_inputs_i(w_i * x_i) + w_0)
# This is the same concept as in Perceptron Learning (see lecture 4_Learning).
# Activation functions Examples: step function (1 if x >= 0 else 0), logistic sigmoid, rectified linear unit (ReLU).
# ANNs can be trained by the Gradient descent algorithm. The weights are updated by repeatedly calculating a gradient
# based on the training data. The Mini-Batch Gradient Descent algorithm computes the gradient based on on a
# few data points to find a compromise between computation cost and accuracy.
# A Neural Network with 1 input and 1 output layer is only capable of learning a linear decision boundary.
# To be able to classify also non-linearly separable data, Multilayer Neural Networks are used.
# Between input- and outputlayer there is at least one hidden layer.
# They are trained with the backpropagation algorithm. It starts with the error of the output unit and then calculating
# the gradient for the weights of the previous layer. This is repeated until the input layer is reached.
# The risk of overfitting is reduced by using the dropout technique. During the learning process, some amount of
# randomly selected units are removed temporarily for each training step.

# Same example as in 4_Learning/0_lecture/banknotes this time solved by an ANN.
# The TensorFlow library is used.

# Read data in from file
with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": 1 if row[4] == "0" else 0
        })

# Separate data into training and testing groups
evidence = [row["evidence"] for row in data]
labels = [row["label"] for row in data]
X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size=0.4
)

# Create a neural network
# keras: API for Tensorflow
# sequential Neural Network: one layer comes after the other
model = tf.keras.models.Sequential()

# Add a hidden layer with 8 units, with ReLU activation.
# densly connected layer: each node of the previous layer is connected to each node in the following layer.
# There are 4 input values in the CSV data, so 4 input units are used for the Neural Network.
model.add(tf.keras.layers.Dense(8, input_shape=(4,), activation="relu"))

# Add output layer with 1 unit, with sigmoid activation
# There is only 1 output needed, because of the binary classification: counterfeit yes, no
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# Train neural network by using the backpropagation algorithm.
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
# epochs: number of times that the Network should be trained on the training data.
model.fit(X_training, y_training, epochs=20)

# Evaluate how well the model performs
model.evaluate(X_testing, y_testing, verbose=2)
