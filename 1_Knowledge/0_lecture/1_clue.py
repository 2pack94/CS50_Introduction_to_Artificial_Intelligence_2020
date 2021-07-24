from logic import *

# In the game of Clue, a murder was committed by a person, using a tool in a location.
# In our model, we mark as True items that we know are related to the murder and False otherwise.

mustard = Symbol("ColMustard")
plum = Symbol("ProfPlum")
scarlet = Symbol("MsScarlet")
characters = [mustard, plum, scarlet]

ballroom = Symbol("ballroom")
kitchen = Symbol("kitchen")
library = Symbol("library")
rooms = [ballroom, kitchen, library]

knife = Symbol("knife")
revolver = Symbol("revolver")
wrench = Symbol("wrench")
weapons = [knife, revolver, wrench]

symbols = characters + rooms + weapons


# Check every symbol for its truth value based on the available knowledge.
# If a symbol is known to be False, don't print it.
# If model check returns False when checking symbol and it also returns False when checking Not(symbol),
# then the KB does not contain enough information to draw an inference.
def checkKnowledge(knowledge):
    for symbol in symbols:
        if modelCheck(knowledge, symbol):
            print(f"{symbol}: YES")
        elif not modelCheck(knowledge, Not(symbol)):
            print(f"{symbol}: MAYBE")


# There must be a person, room, and weapon.
knowledge = And(
    Or(mustard, plum, scarlet),
    Or(ballroom, kitchen, library),
    Or(knife, revolver, wrench)
)

# Initial cards
knowledge.add(And(
    Not(mustard), Not(kitchen), Not(revolver)
))

# Unknown card
knowledge.add(Or(
    Not(scarlet), Not(library), Not(wrench)
))

# Known cards
knowledge.add(Not(plum))
# At this point we can conclude that the murderer is Scarlet.
knowledge.add(Not(ballroom))
# With the additional knowledge of ¬(ballroom) we can deduce that Scarlet committed the murder with a knife in the library.

checkKnowledge(knowledge)
