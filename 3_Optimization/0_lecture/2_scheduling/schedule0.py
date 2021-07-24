"""
Naive backtracking search without any heuristics or inference by maintaining arc consistency.
"""

# Constraint Satisfaction:
# Class of problems where variables need to be assigned values while satisfying some conditions.

# Backtracking Search:
# Search algorithm that takes into account the structure of a constraint satisfaction problem.

# Problem:
# There are 4 students that take the following courses:
# student 1: A, B, C
# student 2: B, D, E
# student 3: C, E, F
# student 4: E, F, G
# Each course needs to have an exam, and the possible days for exams are Monday, Tuesday, and Wednesday.
# The same student can’t have two exams on the same day.

# This problem can be represented as a graph. Each node on the graph is a course.
# Courses are connected by an edge (arc) if they can’t be scheduled on the same day.

VARIABLES = ["A", "B", "C", "D", "E", "F", "G"]
# There are only binary constraints (exams that cannot be on the same day) and no unary constraints
CONSTRAINTS = [
    ("A", "B"),
    ("A", "C"),
    ("B", "C"),
    ("B", "D"),
    ("B", "E"),
    ("C", "E"),
    ("C", "F"),
    ("D", "E"),
    ("E", "F"),
    ("E", "G"),
    ("F", "G")
]


def backtrack(assignment):
    """Runs backtracking search to find an assignment."""

    # Check if assignment is complete (all variables are assigned to a value).
    # Base case. This solution propagates back through the recursion.
    if len(assignment) == len(VARIABLES):
        return assignment

    # Try a new variable
    var = selectUnassignedVariable(assignment)
    # Loop over all values in the domain for this variable
    for value in ["Monday", "Tuesday", "Wednesday"]:
        new_assignment = assignment.copy()
        new_assignment[var] = value
        if consistent(new_assignment):
            result = backtrack(new_assignment)
            if result is not None:
                return result

    # If for a variable all values failed to be assigned, return failure and go back to the previous variable
    # (go up one recursion level). Try to choose a different value for this variable or go back again
    # if failure. If the first variable returns a failure, there is no solution that satisfies the constraints.
    return None


def selectUnassignedVariable(assignment):
    """Chooses a variable not yet assigned, in order."""
    for variable in VARIABLES:
        if variable not in assignment:
            return variable
    return None


def consistent(assignment):
    """Checks to see if an assignment is consistent."""
    for (x, y) in CONSTRAINTS:

        # Only consider arcs where both are assigned
        if x not in assignment or y not in assignment:
            continue

        # If both have same value, then not consistent
        if assignment[x] == assignment[y]:
            return False

    # If nothing inconsistent, then assignment is consistent
    return True


# Start backtracking search with no initial assignments
solution = backtrack(dict())
print(solution)
