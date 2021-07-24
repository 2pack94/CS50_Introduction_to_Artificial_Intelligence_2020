import nltk

# Context-Free Grammar (CFG) only cares about the structure (syntax) of a text regardless of its meaning (semantic).
# The python library nltk (Natural Language Toolkit) is used to analyze the structure of a sentence.

# Terminal symbols: Words (e.g. "country", "and", "before", ...)
# Non-terminal symbols: Define the grammar rules. Generate terminal symbols. (e.g. "N", "NP", "S", ...)

# specify grammar rules
# V: Verb, N: Noun, D: determiner
# NP: Noun Phrase (consists of a noun or determiner + noun)
# VP: Verb Phrase (consists of a verb or verb + Noun Phrase)
# S: Sentence (consists of a Noun Phrase and Verb Phrase)
grammar = nltk.CFG.fromstring("""
    S -> NP VP

    NP -> D N | N
    VP -> V | V NP

    D -> "the" | "a"
    N -> "she" | "city" | "car"
    V -> "saw" | "walked"
""")

parser = nltk.ChartParser(grammar)

sentence = input("Sentence: ").split()
try:
    for tree in parser.parse(sentence):
        tree.pretty_print()
        tree.draw()
except ValueError:
    print("No parse tree possible.")
