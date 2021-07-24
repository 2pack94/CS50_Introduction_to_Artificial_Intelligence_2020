from model import model

# Observed data
observations = [
    "umbrella",
    "umbrella",
    "no umbrella",
    "umbrella",
    "umbrella",
    "umbrella",
    "umbrella",
    "no umbrella",
    "no umbrella"
]

# Predict underlying states (calculate most likely sequence of events given the observations)
predictions = model.predict(observations)
for prediction in predictions:
    print(model.states[prediction].name)
