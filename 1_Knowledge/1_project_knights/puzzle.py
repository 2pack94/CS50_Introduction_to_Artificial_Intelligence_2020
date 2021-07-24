from logic import *

# Puzzle:
# Each character is either a knight or a knave.
# Every statement spoken by a knight is true, and every statement spoken by a knave is false.
# The Goal is to find out which person is a knight or a knave according to the given statements.

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# convenience function to represent XOR logical connective
def XOR(operand1, operand2):
    return And(Or(operand1, operand2), Not(And(operand1, operand2)))

# A person can either be a Knight or a Knave: XOR(Knight, Knave).
# This knowledge represents the game rules and can be applied to the KB of every following Puzzle.
knowledge_rules = And(
    XOR(AKnight, AKnave),
    XOR(BKnight, BKnave),
    XOR(CKnight, CKnave)
)

# Each statement of a person is put into the following construct.
# This represents the rule that the statement is true if a Knight spoke it and false if a Knave spoke it.
# The Biconditional adds the logic that if a statement is true than it was spoken by a Knight and
# if it is false than a Knave spoke it.
def statementA(statement):
    return And(Biconditional(AKnight, statement), Biconditional(AKnave, Not(statement)))

def statementB(statement):
    return And(Biconditional(BKnight, statement), Biconditional(BKnave, Not(statement)))

def statementC(statement):
    return And(Biconditional(CKnight, statement), Biconditional(CKnave, Not(statement)))

# initialize KB for each puzzle
num_puzzles = 4
knowledge_puz = []
for i in range(num_puzzles):
    knowledge_puz.append(
        And(knowledge_rules)
    )

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge_puz[0].add(
    statementA(
        And(AKnight, AKnave)
    )
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge_puz[1].add(
    statementA(
        And(AKnave, BKnave)
    )
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge_puz[2].add(And(
    statementA(
        XOR(And(AKnave, BKnave), And(AKnight, BKnight))
    ),
    statementB(
        XOR(And(AKnave, BKnight), And(AKnight, BKnave))
    )
))

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge_puz[3].add(And(
    XOR(statementA(AKnight), statementA(AKnave)),
    statementB(statementA(AKnave)),
    statementB(CKnave),
    statementC(AKnight)
))


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    for i in range(num_puzzles):
        print(f"Puzzle {i}")
        for symbol in symbols:
            if modelCheck(knowledge_puz[i], symbol):
                print(f"    {symbol}")


if __name__ == "__main__":
    main()

# Solutions:
# Puzzle 0: A is a Knave
# Puzzle 1: A is a Knave, B is a Knight
# Puzzle 2: A is a Knave, B is a Knight
# Puzzle 3: A is a Knight, B is a Knave, C is a Knight
