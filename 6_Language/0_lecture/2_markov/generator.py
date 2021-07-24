import markovify
import sys

# Use a Markov model to generate text. The model is trained on a text.
# Probabilities for every n-th token in an n-gram based on the n words preceding it can be established. 

# Read text from file
if len(sys.argv) != 2:
    sys.exit("Usage: python generator.py sample.txt")
with open(sys.argv[1]) as f:
    text = f.read()

# Train model
text_model = markovify.Text(text)

# Generate sentences
print()
for i in range(5):
    print(text_model.make_sentence())
    print()
