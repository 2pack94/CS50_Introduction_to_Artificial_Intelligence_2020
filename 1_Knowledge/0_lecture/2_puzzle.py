from logic import *

# Logic Puzzle Rules:
# Four different people, Gilderoy, Pomona, Minerva, and Horace, are assigned to four different houses,
# Gryffindor, Hufflepuff, Ravenclaw, and Slytherin. There is exactly one person in each house.

people = ["Gilderoy", "Pomona", "Minerva", "Horace"]
houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

symbols = []

knowledge = And()

# Each of the possible assignments of persons to houses will have to be a proposition in itself.
for person in people:
    for house in houses:
        symbols.append(Symbol(f"{person}{house}"))

# Each person belongs to a house.
for person in people:
    knowledge.add(Or(
        Symbol(f"{person}Gryffindor"),
        Symbol(f"{person}Hufflepuff"),
        Symbol(f"{person}Ravenclaw"),
        Symbol(f"{person}Slytherin")
    ))

# Only one house per person.
for person in people:
    for h1 in houses:
        for h2 in houses:
            if h1 != h2:
                knowledge.add(
                    Implication(Symbol(f"{person}{h1}"), Not(Symbol(f"{person}{h2}")))
                )

# Only one person per house.
for house in houses:
    for p1 in people:
        for p2 in people:
            if p1 != p2:
                knowledge.add(
                    Implication(Symbol(f"{p1}{house}"), Not(Symbol(f"{p2}{house}")))
                )

# Puzzle Information:
# Gilderoy belongs to Gryffindor or Ravenclaw.
# Pomona does not belong to Slytherin.
# Minerva belongs to Gryffindor.
knowledge.add(
    Or(Symbol("GilderoyGryffindor"), Symbol("GilderoyRavenclaw"))
)
knowledge.add(
    Not(Symbol("PomonaSlytherin"))
)
knowledge.add(
    Symbol("MinervaGryffindor")
)

for symbol in symbols:
    if modelCheck(knowledge, symbol):
        print(symbol)
