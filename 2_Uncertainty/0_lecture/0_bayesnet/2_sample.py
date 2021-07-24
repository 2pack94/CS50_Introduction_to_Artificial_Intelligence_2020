import pomegranate
from collections import Counter
from model import model

def generateSample():
    # Each variable is sampled for a value according to its probability distribution.
    # A value for the root node is sampled first. The next node that is dependent on the root node
    # is sampled under the condition of the sampled value for its parent.
    # E.g. if the root node Rain got a sampled value of "none" (picked with probability 0.7),
    # a value from the probability distribution where Rain="none" is picked from node Maintenance.
    # All parent Nodes are always sampled before their child nodes.

    # Map each random variable name in the Network to the generated sample.
    sample = {}

    # Map distribution to the generated sample
    parents = {}

    # Loop over all states (nodes), assuming topological order
    for state in model.states:

        # If non-root node, sample conditional on parents
        if isinstance(state.distribution, pomegranate.ConditionalProbabilityTable):
            sample[state.name] = state.distribution.sample(parent_values=parents)

        # Otherwise, just sample from the distribution alone
        else:
            sample[state.name] = state.distribution.sample()

        # Keep track of the sampled value in the parents mapping
        parents[state.distribution] = sample[state.name]

    # Return generated sample
    return sample

# Rejection sampling
# Compute distribution of Appointment given that train is delayed
# Reject all samles where train is not delayed
N = 10000
data = []
for i in range(N):
    sample = generateSample()
    if sample["train"] == "delayed":
        data.append(sample["appointment"])
# count number of occurrences for each value of the "Appointment" variable.
counter = Counter(data)
# divide number of occurrences by total number of samples to get approximated probability.
for value in counter:
    print(f"{value}: {counter[value] / len(data)}")
