import nltk

# Wordnet is a database similar to a dictionary. It is built into the nltk library.

nltk.download('wordnet')

word = input("Word: ")
synsets = nltk.corpus.wordnet.synsets(word)

for synset in synsets:
    print()
    print(f"{synset.name()}: {synset.definition()}")
    for hypernym in synset.hypernyms():
        print(f"  {hypernym.name()}")
