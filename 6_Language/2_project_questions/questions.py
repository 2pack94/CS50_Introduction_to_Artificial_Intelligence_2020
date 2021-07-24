import nltk
import math
import string
import sys
import os

# Simple question answering system based on inverse document frequency.
# Find sentences from the files in the corpus that are relevant to a user’s query.
# The most relevant documents are determined with TF-IDF and the most relevant sentences within
# these documents are determined with a combination of IDF and a query term density measure.

# example queries:
# What are the types of supervised learning?
# When was Python 3.0 released?
# How do neurons connect in a neural network?

# number of files that should be matched for any given query.
FILE_MATCHES = 1
# number of sentences that should be matched for any given query.
SENTENCE_MATCHES = 3

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Download required nltk packages into "~/nltk_data/"
    nltk.download('punkt')
    nltk.download('stopwords')

    # Calculate IDF values across files
    files = loadFiles(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = computeIdfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = topFiles(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = computeIdfs(sentences)

    # Determine top sentence matches
    matches = topSentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def loadFiles(directory):
    """
    Returns a dictionary mapping the filename of each `.txt` file inside the given directory
    to the file's contents as a string.
    """
    files = dict()
    for file_name in os.listdir(directory):
        with open(os.path.join(directory, file_name), "r") as fp:
            files[file_name] = fp.read()
    return files


def tokenize(document):
    """
    Given a document (represented as a string), returns a list of all of the
    words in that document, in order.
    The document is processed by coverting all words to lowercase, and removing any
    punctuation or English stopwords. Stopwords are the most common, short function words.
    """
    words = nltk.word_tokenize(document)
    words = list(map(lambda x: x.lower(), words))
    # remove words that only consist of punctuation.
    for i in range(len(words)):
        is_word = False
        for char in words[i]:
            if char not in string.punctuation:
                is_word = True
                break
        if not is_word:
            words[i] = ""
    stopwords = nltk.corpus.stopwords.words("english")

    return [word for word in words if word and word not in stopwords]


def computeIdfs(documents):
    """
    Given a dictionary that maps names of documents to a list of words,
    returns a dictionary that maps words to their IDF values.
    Any word that appears in at least one of the documents will be in the resulting dictionary.
    """
    idfs = dict()
    words = set()
    for doc_words in documents.values():
        words.update(doc_words)
    
    num_documents = len(documents)
    for word in words:
        num_occurrences = 0
        for doc_name in documents:
            if word in documents[doc_name]:
                num_occurrences += 1
        idfs[word] = math.log(num_documents / num_occurrences)

    return idfs


def topFiles(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of files to a list
    of their words), and `idfs` (a dictionary mapping words to their IDF values),
    returns a list of the filenames of the the `n` top files that match the query,
    ranked according to tf-idf.
    """
    # For each file, sum up the tf-idf for every word in the query
    tfidfs = dict()
    for file in files:
        tfidfs[file] = 0
        for word in query:
            tf = files[file].count(word)
            tfidfs[file] += tf * idfs.get(word, 0)
    
    tfidfs_list = sorted(tfidfs, key = lambda x: tfidfs[x], reverse=True)
    return tfidfs_list[:n]


def topSentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping sentences to a list
    of their words), and `idfs` (a dictionary mapping words to their IDF values),
    returns a list of the `n` top sentences that match the query, ranked according to idf.
    If there are ties, preference is given to sentences that have a higher query term density.
    """
    # For each sentence, sum up the idf for every word in the query if the word appears in the sentence.
    query_idfs = dict()
    for sentence in sentences:
        query_idfs[sentence] = 0
        for word in query:
            if word in sentences[sentence]:
                query_idfs[sentence] += idfs.get(word, 0)

    # Query term density is the proportion of words in the sentence that are also words in the query.
    query_term_densities = dict()
    for sentence in sentences:
        query_tf = 0
        for word in query:
            query_tf += sentences[sentence].count(word)
        query_term_densities[sentence] = query_tf / len(sentences[sentence])

    # Order the sentences primarily after the idf and secondarily after the query term density.
    # Because python sorts are stable, the sentences can first be sorted after
    # the query term density and then after the idf.
    sentences_ordered = query_idfs.keys()
    sentences_ordered = sorted(sentences_ordered, key = lambda x: query_term_densities[x], reverse=True)
    sentences_ordered = sorted(sentences_ordered, key = lambda x: query_idfs[x], reverse=True)

    return sentences_ordered[:n]


if __name__ == "__main__":
    main()
