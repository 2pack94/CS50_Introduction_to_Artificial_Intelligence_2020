import math
import nltk
import os
import sys

# Use topic modeling to discover the topics for a set of documents.
# Get the term frequency (TF) for every term in a corpus of documents.
# The Problem is that the most frequent words are function words that are used for syntactic purposes,
# but don't have any meaning by themselfs.

def main():
    """Calculate top term frequencies for a corpus of documents."""

    if len(sys.argv) != 2:
        sys.exit("Usage: python tfidf.py corpus")
    
    # Download Tokenizer package into "~/nltk_data/"
    nltk.download('punkt')
    print("Loading data...")
    corpus = load_data(sys.argv[1])

    # Get all words in corpus
    print("Extracting words from corpus...")
    words = set()
    for filename in corpus:
        words.update(corpus[filename])

    # Calculate TFs
    print("Calculating term frequencies...")
    tfs = dict()
    for filename in corpus:
        tfs[filename] = []
        for word in corpus[filename]:
            tf = corpus[filename][word]
            tfs[filename].append((word, tf))

    # Sort and get top 5 term frequencies for each file
    print("Computing top terms...")
    for filename in corpus:
        tfs[filename].sort(key=lambda tfidf: tfidf[1], reverse=True)
        tfs[filename] = tfs[filename][:5]

    # Print results
    print()
    for filename in corpus:
        print(filename)
        for term, score in tfs[filename]:
            print(f"    {term}: {score:.4f}")


def load_data(directory):
    files = dict()
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as f:

            # Extract words
            contents = [
                word.lower() for word in
                nltk.word_tokenize(f.read())
                if word.isalpha()
            ]

            # Count frequencies
            frequencies = dict()
            for word in contents:
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
            files[filename] = frequencies

    return files


if __name__ == "__main__":
    main()
