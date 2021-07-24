from model import model

# Calculate the probability distribution for each node in the Network.
# The function parameter is a dict containing the evidence variables (variables that have been observed).
predictions = model.predict_proba({
    "train": "delayed"
})

# Print predictions for each node
for node, prediction in zip(model.states, predictions):
    if isinstance(prediction, str):
        print(f"{node.name}: {prediction}")
    else:
        print(f"{node.name}")
        for value, probability in prediction.parameters[0].items():
            print(f"    {value}: {probability:.4f}")

# Inference by enumeration:
# Finding the probability distribution of variable X (query variable)
# given observed evidence e and some hidden variables Y.
# The hidden variables Y are all variables that are not X and not in e.
# Convert P(X | e) to a joint probability:
# P(X | e) = P(X, e) / P(e) = αP(X, e)
# α: proportion constant resulting from 1 / P(e)
# Use marginalization to calculate P(X, e):
# Sum up P(X, e, y) where y is each time a different value of the hidden variables Y
# (each hidden variable is enumerated). (Each P(X, e, y) is calculated like in 0_likelihood.py)
# To get the probability distribution of variable X use this formula for every value X = x_i
# The resulting probability distribution is multiplied by α in a way that it sums up to 1 (normalized).
