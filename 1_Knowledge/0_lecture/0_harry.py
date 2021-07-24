from logic import *

# Create new classes, each having a name, or a symbol, representing each proposition.
rain = Symbol("rain")               # It is raining.
hagrid = Symbol("hagrid")           # Harry visited Hagrid.
dumbledore = Symbol("dumbledore")   # Harry visited Dumbledore.

# Save sentences into the Knowledge Base (KB)
knowledge = And(
    Implication(Not(rain), hagrid), # ¬(It is raining) → (Harry visited Hagrid)
    Or(hagrid, dumbledore),         # (Harry visited Hagrid) ∨ (Harry visited Dumbledore).
    Not(And(hagrid, dumbledore)),   # ¬(Harry visited Hagrid ∧ Harry visited Dumbledore) i.e. Harry did not visit both Hagrid and Dumbledore.
    dumbledore                      # Harry visited Dumbledore.
)

# The query is: Is it raining?
# The Model Checking algorithm is used to find out if the query is entailed by the KB.
query = rain
print(modelCheck(knowledge, query))

# If the KB does not contain enough information to conclude the truth value of a query, model check will return False.
# Example:
# knowledge = dumbledore
# query = rain
# The enumerated models would look like this:
# dumbledore    rain    KB
# ------------------------------
# False         False   False
# False         True    False
# True          False   True
# True          True    True
# For the model dumbledore -> True and rain -> False, the KB is true, but the query is False.
# If the query would have been: query = Not(rain)
# Then for the model dumbledore -> True and rain -> True, the KB is true, but the query is False.
