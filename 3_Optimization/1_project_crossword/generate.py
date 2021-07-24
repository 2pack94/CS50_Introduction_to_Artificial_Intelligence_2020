import sys
from crossword import *

# Problem:
# Solve a Crossword Puzzle given its structure (defined in a structure file)
# and a list of available words (defined in a words file).

# The problem can be modeled as a Constraint Satisfaction Problem (CSP).
# Each sequence of cells is one variable. The Domain of each variable is a list of
# possible words (values) that can be filled in that cell sequence.
# The unary constraint on a variable is given by its length.
# The binary constraints on a variable are given by its overlap with neighboring variables.
# Neighboring variables share a single cell that is common to them both.
# An overlap is represented with the tuple (i, j), where v1's ith character overlaps v2's jth character.
# Another binary constraint is that two variables cannot have the same value.

# The problem will be solved by using backtracking search with inference (Maintaining Arc-Consistency).
# Before starting the backtracking search, Node- and Arc-Consistency will be enforced.
# Variables will be selected primarily after the Minimum Remaining Values (MRV) heuristic
# and secondarily after the Degree heuristic.
# Values will be selected after the Least-Constraining Values heuristic.

class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        # Crossword object
        self.crossword = crossword
        # Initialize the domain for each variable as the full word list
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letterGrid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letterGrid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letterGrid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforceNodeConsistency()
        if not self.ac3():
            return None
        return self.backtrack(dict())

    def enforceNodeConsistency(self):
        """
        Updates `self.domains` such that each variable is node-consistent.
        Removes any values that are inconsistent with a variable's unary constraints;
        in this case, the length of the word.
        """
        for var in self.crossword.variables:
            self.domains[var] = set(value for value in self.domains[var] if len(value) == var.length)

    def revise(self, var1, var2, inferences_domain=None):
        """
        Make variable var1 arc consistent with variable var2.
        Removes values from self.domains[var1] for which there is no possible corresponding value
        for var2 in self.domains[var2] that doesn't cause a conflict.
        A conflict is a cell for which two variables disagree on what character value it should take on.
        Returns True if a revision was made to the domain of var1; return False otherwise.
        If inferences_domain (dict) is specified,
        it will contain a set of the removed values for each variable (key).
        """
        is_revised = False
        overlap = self.crossword.overlaps[var1, var2]
        if not overlap:
            return is_revised
        remove_values = set()
        for word1 in self.domains[var1]:
            is_conflict = True
            for word2 in self.domains[var2]:
                # There is no conflict if both words are different (same words are not allowed) and
                # they both have the same character for the overlapping cell.
                if word1 != word2 and word1[overlap[0]] == word2[overlap[1]]:
                    is_conflict = False
                    break
            if is_conflict:
                remove_values.add(word1)

        # When used within ac3, revise() may be invoked multiple times to revise the same Variable.
        for word in remove_values:
            is_revised = True
            self.domains[var1].remove(word)

            if inferences_domain:
                inferences_domain[var1].add(word)

        return is_revised

    def ac3(self, arcs=None, assignment=None, inferences_domain=None):
        """
        Updates `self.domains` such that each variable is arc consistent,
        ensuring that binary constraints are satisfied.
        If `arcs` is None, all arcs in the problem are made consistent.
        Otherwise, `arcs` is used as the initial list of arcs to make consistent.
        Each arc is a tuple (x, y) of a variable x and a different variable y.
        Returns True if arc consistency is enforced and no domains are empty;
        returns False if one or more domains end up empty.
        `assignment` and `inferences_domain` are used for maintaining arc-consistency
        inside backtracking search (see inference function).
        """

        # Use list as queue where items get enqueued at the end and dequeued at the beginning.
        queue = []
        if not arcs:
            # Enqueue all arcs
            for var in self.crossword.variables:
                neighbors = self.crossword.neighbors(var)
                for neighbor in neighbors:
                    queue.append((var, neighbor))
        else:
            queue = arcs

        while queue:
            (var1, var2) = queue.pop(0)
            if self.revise(var1, var2, inferences_domain):
                if len(self.domains[var1]) == 0:
                    # CSP is unsolvable
                    return False
                # since var1's domain was changed, all associated arcs must be checked for consistency.
                # arc (var2, var1) does not need to be checked, because for the revision of var2's domain
                # it doesn't matter if the removed incompatible values from var1's domain are there or not.
                for neighbor in self.crossword.neighbors(var1) - set([var2]):
                    # Do not revise a Variable that is already assigned.
                    if assignment and neighbor in assignment:
                        continue
                    arc = (neighbor, var1)
                    if arc not in queue:
                        queue.append(arc)

        return True

    def inference(self, assignment, var):
        """
        Called in the backtrack function after assigning a variable to a value to maintain arc-consistency
        and to make inferences about additional assignments that can be made.
        `var` is the Variable that was just assigned. `assignment` is the current assignment dict.
        Returns `inferences` and `inferences_domain`. `inferences_domain` is a dict that maps variables (keys)
        to a set of values that got removed from the domain of the variable during this function call.
        `inferences` is a dict that maps variables (keys) to values. The content of this dict can be added to
        `assignment` to speed up the assignment process.
        The changes that this function does to the domain of variables must be reverted again if there is a
        backtrack (when an assignment got reverted).
        """
        inferences = dict()
        inferences_domain = {
            var_inf: set() for var_inf in self.crossword.variables
        }
        # For the currently assigned variable, set its domain to the assigned value.
        # Only then ac3 can find new values to be removed from other domains.
        inferences_domain[var] = self.domains[var] - set([assignment[var]])
        self.domains[var] = set([assignment[var]])

        # It's optional to remove the assigned value from the domain of all other variables.
        # Because duplicated words are restricted in all other parts of the program anyways.

        # Get arcs that ac3 should be initialized with. The assignment of a variable only initially impacts
        # its neighbors that are not assigned yet (ac3 will further enqueue others that are impacted).
        arcs = []
        for neighbor in self.crossword.neighbors(var):
            if neighbor not in assignment:
                arcs.append((neighbor, var))

        # If ac3 returns False, the next recursion level of backtrack will use the variable that has
        # an empty domain (because of MRV heuristic). Since this variable cannot have an assignment,
        # backtrack goes back one recursion level.
        if self.ac3(arcs, assignment, inferences_domain):
            # populate inferences dict.
            # An inference can be made if the domain of an unassigned variable has only 1 value.
            for var_inf in self.crossword.variables:
                if var_inf not in assignment and len(self.domains[var_inf]) == 1:
                    inferences[var_inf] = list(self.domains[var_inf])[0]

        return inferences, inferences_domain

    def assignmentComplete(self, assignment):
        """
        Returns True if `assignment` is complete, False otherwise.
        An Assignment is complete if every variable is assigned to a value.
        """
        for var in self.crossword.variables:
            if not assignment.get(var):
                return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent, False otherwise.
        I.e. all currently assigned words fit in crossword puzzle without conflicting characters.
        """
        values_assigned = []
        for var in assignment:
            value = assignment[var]
            # words should fit in the sequence of cells
            if len(value) != var.length:
                return False
            # words should not be duplicated
            if value in values_assigned:
                return False
            values_assigned.append(value)
            # Neighboring variables should not have conflicts.
            # (Neighbors are checked twice here, 1 time in each direction, which is a bit inefficient)
            for neighbor in self.crossword.neighbors(var):
                neighbor_value = assignment.get(neighbor)
                if neighbor_value:
                    overlap = self.crossword.overlaps[var, neighbor]
                    if value[overlap[0]] != neighbor_value[overlap[1]]:
                        return False
        return True

    def orderedDomainValues(self, var, assignment):
        """
        Returns a list of values in the domain of `var`, ordered according to the
        least-constraining values heuristic. Values that constrain the least amount
        of values for neighboring unassigned variables are placed first.
        """
        # For each value in the domain of var, count the number of conflicting values
        # in the domain of every unassigned neighbor.
        values_constraints = {
            val: 0 for val in self.domains[var]
        }

        for value in self.domains[var]:
            for neighbor in self.crossword.neighbors(var):
                if neighbor not in assignment:
                    overlap = self.crossword.overlaps[var, neighbor]
                    for neighbor_value in self.domains[neighbor]:
                        # There is a conflict if both words are the same or
                        # they both have a different character for the overlapping cell.
                        if value == neighbor_value or value[overlap[0]] != neighbor_value[overlap[1]]:
                            values_constraints[value] += 1
        
        # Sort the values after the number of conflicts in ascending order.
        ordered_domain = sorted(values_constraints, key = lambda x: values_constraints[x])
        return ordered_domain

    def selectUnassignedVariable(self, assignment):
        """
        Returns an unassigned variable not already part of `assignment`.
        All variables are ordered primarily after the Minimum Remaining Values (MRV) heuristic
        and secondarily after the Degree heuristic.
        MRV heuristic: Choose the variable with the least amount of values in its domain.
        The idea is that if a variables domain was restricted by enforcing arc-consistency,
        its better to assign early to reduce backtracks.
        Degree heuristic: Choose the variable with the most unassigned neighbors.
        The goal is to constrain many other variables to speed up the algorithm.
        """
        # construct a list that contains the following tuples:
        # (Variable object, number of values in Variable domain, number of unassigned neighbors for the Variable)
        vars = []
        for var in self.crossword.variables:
            if var not in assignment:
                vars.append(
                    (var, len(self.domains[var]), len(self.crossword.neighbors(var) - set(assignment)))
                )

        # First order after the number of values in Variable domain, ascending order.
        ordered_vars = sorted(vars, key = lambda x: x[1])
        num_tied = 0
        for var in ordered_vars:
            num_tied += 1
            if var[1] != ordered_vars[0][1]:
                break
        
        # Take the Variables that have the same number of values in their domain.
        # Order after the number of neighbors, descending order
        ordered_vars = ordered_vars[:num_tied]
        ordered_vars = sorted(ordered_vars, key = lambda x: x[2], reverse=True)

        # Select the Variable with the least amount of values in its domain and with the most neighbors.
        return ordered_vars[0][0]

    def backtrack(self, assignment):
        """
        Runs Backtracking Search on a partial assignment for the
        crossword and returns a complete assignment if possible to do so.
        If no full assignment is possible, returns None.
        `assignment` is a dict that maps variables (keys) to words (values).
        """
        if len(assignment) == len(self.crossword.variables):
            return assignment

        var = self.selectUnassignedVariable(assignment)
        for value in self.orderedDomainValues(var, assignment):
            assignment[var] = value
            if self.consistent(assignment):
                # apply inferences obtained by maintaining arc-consistency with ac3.
                inferences, inferences_domain = self.inference(assignment, var)
                assignment.update(inferences)
                result = self.backtrack(assignment)
                if result is not None:
                    return result

                # Remove inferences again, if there was a backtrack (if went up recursion level).
                for var_inf in inferences:
                    assignment.pop(var_inf)
                for var_inf in inferences_domain:
                    self.domains[var_inf].update(inferences_domain[var_inf])

            # Remove assignment if the value didn't work out
            assignment.pop(var)

        return None

def main():

    # Check usage
    # Example:
    # $ python3 generate.py data/structure0.txt data/words0.txt output.png
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)
    else:
        print("No solution.")



if __name__ == "__main__":
    main()
