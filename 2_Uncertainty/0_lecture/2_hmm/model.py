from pomegranate import *

# A hidden Markov model is a type of a Markov model for a system with hidden states
# that generate some observed event.

# Example:
# An AI wants to infer the weather (the hidden state), but it only has access to an indoor camera
# that records how many people brought umbrellas with them.

# Observation model for each state (also called sensor model or emission model)
sun = DiscreteDistribution({
    "umbrella": 0.2,
    "no umbrella": 0.8
})

rain = DiscreteDistribution({
    "umbrella": 0.9,
    "no umbrella": 0.1
})

states = [sun, rain]

# Transition model
transitions = numpy.array(
    [[0.8, 0.2], # Tomorrow's predictions if today = sun
     [0.3, 0.7]] # Tomorrow's predictions if today = rain
)

# Starting probabilities
starts = numpy.array([0.5, 0.5])

# Create the model
model = HiddenMarkovModel.from_matrix(
    transitions, states, starts,
    state_names=["sun", "rain"]
)
model.bake()
