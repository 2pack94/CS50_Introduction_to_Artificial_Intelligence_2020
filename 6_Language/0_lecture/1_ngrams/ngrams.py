from collections import Counter
import math
import nltk
import os
import sys

# An n-gram is a sequence of n items from a sample of text. In a character n-gram, the items are characters,
# and in a word n-gram the items are words. A unigram, bigram, and trigram are sequences of one, two, and three items.
# An AI can use n-grams extracted from a text to predict the next word while a person is typing a text.
# With the knowledge of trigrams and given an input of 2 words, a probability distribution for which word will
# likely follow can be calculated.

# To extract n-grams from a text Tokenization is needed.
# Tokenization is the task of splitting a text into pieces (tokens). Tokens can be words as well as sentences.
# A Tokenizer implements proper handling for punctuation, hypens, apostrophes, etc.

def main():
    """Calculate top term frequencies for a corpus of documents."""

    if len(sys.argv) != 3:
        sys.exit("Usage: python ngrams.py n corpus")

    # Download Tokenizer package into "~/nltk_data/"
    nltk.download('punkt')

    print("Loading data...")

    n = int(sys.argv[1])
    corpus = load_data(sys.argv[2])

    # Compute n-grams
    ngrams = Counter(nltk.ngrams(corpus, n))

    # Print most common n-grams
    for ngram, freq in ngrams.most_common(10):
        print(f"{freq}: {ngram}")


def load_data(directory):
    contents = []

    # Read all files and extract words
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as f:
            contents.extend([
                word.lower() for word in
                nltk.word_tokenize(f.read())
                if any(c.isalpha() for c in word)
            ])
    return contents


if __name__ == "__main__":
    main()
