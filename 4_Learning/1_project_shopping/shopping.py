import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# The csv file contains a dataset of shopping behavior on a shopping website.
# The input data are different metrics of user behavior and the output is if the user made a purchase or not.
# A k-nearest neighbor Classifier (with k=1) will be used to predict whether or not the user will make a purchase.
# Instead of using the total number of correct prediction to evaluate the accuracy of the system,
# the sensitivity (true positive rate) and specificity (true negative rate) are measured.

# proportion of the testing data set
TEST_SIZE = 0.4

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = loadData(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = trainModel(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def parseEvidenceMonth(month):
    """This function parses the column `Month` in the evidence data in the CSV file"""
    index = 0
    months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    try:
        index = months.index(month)
    except ValueError:
        pass

    return index

def loadData(filename):
    """
    Loads shopping data from the CSV file `filename` and converts it into a list of
    evidence lists and a list of labels. Returns a tuple (evidence, labels).
    evidence is a list of lists, where each list contains the parsed values from a row in the CSV file.
    Each label is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []

    with open(filename) as fp:
        reader = csv.reader(fp)
        # skip the table header
        next(reader)

        # This list contains a parsing function for every evidence column in the CSV file, in order.
        evidence_types = [
            int,                                        # Administrative, an integer
            float,                                      # Administrative_Duration, a floating point number
            int,                                        # Informational, an integer
            float,                                      # Informational_Duration, a floating point number
            int,                                        # ProductRelated, an integer
            float,                                      # ProductRelated_Duration, a floating point number
            float,                                      # BounceRates, a floating point number
            float,                                      # ExitRates, a floating point number
            float,                                      # PageValues, a floating point number
            float,                                      # SpecialDay, a floating point number
            parseEvidenceMonth,                         # Month, an index from 0 (January) to 11 (December)
            int,                                        # OperatingSystems, an integer
            int,                                        # Browser, an integer
            int,                                        # Region, an integer
            int,                                        # TrafficType, an integer
            lambda x: 0 if x == "New_Visitor" else 1,   # VisitorType, an integer 0 (not returning) or 1 (returning)
            lambda x: 0 if x == "FALSE" else 1,         # Weekend, an integer 0 (if false) or 1 (if true)
        ]
        # The last column is the label (output).
        for row in reader:
            evidence.append(
                [evidence_types[i](row[i]) for i in range(len(row[:-1]))]
            )
            labels.append(
                0 if row[-1] == "FALSE" else 1
            )
    
    return (evidence, labels)

def trainModel(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, returns a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    """
    Given a list of actual labels from the test set and a list of predicted labels,
    returns the tuple (sensitivity, specificity).
    A label is either a 1 (positive) or 0 (negative).
    `sensitivity` is a float value from 0 to 1 representing the "true positive rate":
    the proportion of actual positive labels that were accurately identified.
    `specificity` is a float value from 0 to 1 representing the "true negative rate":
    the proportion of actual negative labels that were accurately identified.
    """
    true_positives = 0
    true_negatives = 0
    positives = 0
    negatives = 0
    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            positives += 1
            if actual == predicted:
                true_positives += 1
        else:
            negatives += 1
            if actual == predicted:
                true_negatives += 1
    
    sensitivity = true_positives / positives
    specificity = true_negatives / negatives
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
