import nltk
import re
import sys

# Context-free Grammar
# Write a parser that is able to parse all of the sentences inside the sentences folder.

# terminal symbols
TERMINALS = """
    Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
    Adv -> "down" | "here" | "never"
    Conj -> "and" | "until"
    Det -> "a" | "an" | "his" | "my" | "the"
    N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
    N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
    N -> "smile" | "thursday" | "walk" | "we" | "word"
    P -> "at" | "before" | "in" | "of" | "on" | "to"
    V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
    V -> "smiled" | "tell" | "were"
"""

# In most cases grammar loops should be avoided (e.g. "NP -> Det NP" would allow for infinite determiners).
# Sometimes loops can be used intentional (e.g. "AdjP -> Adj AdjP" allows for infinite adjectives).
# An Adverb can occur before or after a Verb Phrase.

# non-terminal symbols
NONTERMINALS = """
    S -> NP VP | S PP | S Conj S | S Conj VP

    AdjP -> Adj | Adj AdjP
    NP -> N | Det N | AdjP N | Det AdjP N
    PP -> P NP
    AdvP -> Adv V
    VP -> V | V NP | V PP | VP Adv
    VP -> AdvP | AdvP NP | AdvP PP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)

def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            sentence = f.read()
    # Otherwise, get sentence as input
    else:
        sentence = input("Sentence: ")
    
    # Download Tokenizer package into "~/nltk_data/"
    nltk.download('punkt')

    # Convert input into list of words
    sentence = preprocess(sentence)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(sentence))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in npChunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    The sentence will be Pre-processed by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic character.
    """
    word_list = []
    for word in nltk.word_tokenize(sentence):
        word_processed = word.lower()
        match_obj = re.search(r'[a-z]', word_processed)
        if not bool(match_obj):
            continue
        word_list.append(word_processed)

    return word_list


def npChunk(tree):
    """
    Returns a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence whose label is "NP"
    that does not itself contain any other noun phrases as subtrees.
    nltk tree documentation: https://www.nltk.org/_modules/nltk/tree.html
    """
    np_chunks = []
    # If a subtree with the label "NP", has only 1 subtree that has the label "NP" (i.e. the subtree itself),
    # then this subtree does not contain any other noun phrases and is therefore a noun phrase chunk.
    # (According to the current definition of NP, a NP cannot contain another NP anyways.)
    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            num_np_labels = 0
            for np_subtree in subtree.subtrees():
                if np_subtree.label() == "NP":
                    num_np_labels += 1
            if num_np_labels == 1:
                np_chunks.append(subtree)

    return np_chunks


if __name__ == "__main__":
    main()
