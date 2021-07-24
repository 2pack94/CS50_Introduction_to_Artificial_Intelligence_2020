import math
import nltk
import os
import sys

# Use a list of function words to exclude them from counting, so only the content words are counted.
# The Problem is that common words that occur in all documents are listed for every document.
# This however does not hold any useful information for classifying the documents.

def main():
    """
    Calculate top term frequencies for a corpus of documents.
    Excludes stop words.
    """

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

    with open("function_words.txt") as f:
        function_words = set(f.read().splitlines())

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

                if word in function_words:
                    continue
                elif word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
            files[filename] = frequencies

    return files


if __name__ == "__main__":
    main()
