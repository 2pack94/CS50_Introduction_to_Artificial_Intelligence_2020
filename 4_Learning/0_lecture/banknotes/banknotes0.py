import csv
import random

from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# The csv file contains a dataset of counterfeit banknotes.
# Measurements of banknote properties are mapped to the value 0 (not counterfeit) or 1 (counterfeit).
# The dataset will be split into a training and testing set (Holdout Cross Validation).
# The python library scikit-learn provides different classification algorithms.
# Classification is a task where the function maps an input (banknote properties)
# to a discrete output (counterfeit yes, no).
# The algorithms can be compared by seeing how many correct predictions can be made on the testing data
# after conditioning the model to the training data.

# k-nearest-neighbors classification:
# A new sample is classified based on the label of the k nearest neighbors.
# The nearest neighbors are the data points that have the smallest distance to the test point.
# k is an integer that can be chosen by the programmer.

# Perceptron Learning:
# Creates a linear decision boundary between two classes of data.
# A decision boundary with n inputs (dimensions) forms a hyperplane.
# The hypothesis function that is predicting the output is a linear equation consisting of weights
# and the variables for the inputs.
# The training phase updates the weights to create a threshold that best separates the data.

# Support Vector Machine (Support Vector Classifier, SVC):
# Create a boundary that is as far as possible from the two groups it separates (Maximum Margin Separator).
# Can represent decision boundaries between more than two outputs, as well as non-linear decision boundaries.

models = []
models.append(Perceptron())
models.append(svm.SVC())
models.append(KNeighborsClassifier(n_neighbors=1))
# Gaussian Naive Bayes
models.append(GaussianNB())

# Read data in from file
with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    # The first 4 columns are the evidence (input) and the last column is the label (output).
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": "Authentic" if row[4] == "0" else "Counterfeit"
        })

for model in models:
    # Separate data into training and testing groups
    holdout = int(0.4 * len(data))
    random.shuffle(data)
    testing = data[:holdout]
    training = data[holdout:]

    # Train model on training set
    X_training = [row["evidence"] for row in training]
    y_training = [row["label"] for row in training]
    model.fit(X_training, y_training)

    # Make predictions on the testing set
    X_testing = [row["evidence"] for row in testing]
    y_testing = [row["label"] for row in testing]
    predictions = model.predict(X_testing)

    # Compute how accurate the model is.
    correct = 0
    incorrect = 0
    total = 0
    for actual, predicted in zip(y_testing, predictions):
        total += 1
        if actual == predicted:
            correct += 1
        else:
            incorrect += 1

    # Print results
    print(f"Results for model {type(model).__name__}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy: {100 * correct / total:.2f}%")
    print()
