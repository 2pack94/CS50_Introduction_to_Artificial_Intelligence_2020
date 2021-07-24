from model import model

# Calculate probability for a given observation
# No Rain, no Maintenance, Train on time, attend the meeting
probability = model.probability([["none", "no", "on time", "attend"]])
print(probability)

probability = model.probability([["light", "no", "delayed", "miss"]])
print(probability)

# Calculation of joint probability:
# P(light, no, delayed, miss) = P(light) * P(no | light) * P(delayed | light, no) * P(miss | delayed)
#                             = 0.2 * 0.8 * 0.3 * 0.4 = 0.0192
# The value of each of the individual probabilities can be found in
# the probability distributions defined in the Nodes.
# Note that even though the Train is dependent on Rain and Maintenance,
# the Meeting is only dependent on the Train.
