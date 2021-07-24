import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # self.board contains True for every cell that has a mine, False for all other cells.
        # self.board_revealed contains the numbers of the cells that were already revealed.
        self.board = []
        self.board_revealed = []
        for i in range(self.height):
            self.board.append([])
            self.board_revealed.append([])
            for j in range(self.width):
                self.board[i].append(False)
                self.board_revealed[i].append(None)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

    def print(self):
        """
        Prints a text-based representation of where mines are located
        and which cells have been revealed.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                elif self.board_revealed[i][j] is not None:
                    print(f"|{self.board_revealed[i][j]}", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def isMine(self, cell):
        i, j = cell
        return self.board[i][j]

    def numNearbyMines(self, cell):
        """
        Returns the number of mines that are within one row and column of a given cell,
        not including the cell itself.
        This is the number displayed inside the cell.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count
    
    def reveal(self, cell):
        if self.isMine(cell):
            return
        self.board_revealed[cell[0]][cell[1]] = self.numNearbyMines(cell)

    def isWon(self):
        """
        Returns True if the game is won (if all cells were revealed).
        """
        num_revealed = 0
        num_total = self.height * self.width - len(self.mines)
        for i in range(self.height):
            for j in range(self.width):
                if self.board_revealed[i][j] is not None:
                    num_revealed += 1
        if num_revealed >= num_total:
            return True
        return False


class Sentence():
    """
    Logical statement about a Minesweeper game.
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    Each revealed number on the board gets assigned a Sentence object
    containing the adjacent cells that still have unknown content.
    Sentences can also be inferred from other Sentences when one Sentence
    contains a subset of cells from another Sentence.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"
    
    def isEmpty(self):
        """
        Returns True if there are no cells left in the Sentence
        (when every cell was marked as mine or safe)
        """
        return not bool(self.cells)

    def knownMines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if len(self.cells) == self.count:
            return self.cells
        return set()

    def knownSafes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return self.cells
        return set()

    def markMine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def markSafe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """
    # Notes on Python sets:
    # Sets do not duplicate values so its unproblematic to add values.
    # (set1 - set2) will return a new set with elements in set1 that are not in set2.
    # Modifying a set while iterating over it doesn't work.
    # Sentence objects cannot be put a in set because they are not hashable.
    # An empty set evaluated as boolean returns False.

    def __init__(self, height=8, width=8):
        # Set initial height and width of the Minesweeper board
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines.
        # Cells in self.safes that are not yet in self.moves_made can be considered as next safe moves.
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true.
        self.knowledge = []

    def addSentence(self, cell, count):
        """
        Create a valid Sentence object from a cell and its number of neighboring mines.
        Add the Sentence to the knowledge.
        """
        # To create a valid sentence it must be created based on the current knowledge
        # about safe cells and mines, because the sentences will only be updated on any new knowledge.
        # Loop over all adjacent cells.
        cells = set()
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                cell_loop = (i, j)
                # Ignore the cell itself, out of bounds cells and known safe cells.
                if (cell_loop == cell or
                    i >= self.height or i < 0 or j >= self.width or j < 0 or
                    cell_loop in self.safes
                ):
                    continue
                # Decrease the mine count for every known mine.
                if cell_loop in self.mines:
                    count -= 1
                    continue
                cells.add(cell_loop)

        if cells:
            sentence = Sentence(cells, count)
            if sentence not in self.knowledge:
                self.knowledge.append(sentence)

    def inferSentences(self):
        """
        Infer new sentences from the current knowledge.
        Any time there are two sentences (cells1, count1) and (cells2, count2) where cells1 is a subset of cells2,
        then a new sentence (cells2 - cells1, count2 - count1) can be constructed.
        A subset of cells can only contain equal or less mines.
        """
        # First check all sentences in self.knowledge against all other sentences.
        # All inferred sentences are again checked against all other sentences
        # until no sentence can be inferred any more.
        check_sentences = self.knowledge
        while True:
            inferred_sentences = []
            for sentence in check_sentences:
                for sentence_comp in self.knowledge:
                    if (len(sentence.cells) > len(sentence_comp.cells) and
                        sentence_comp.cells.issubset(sentence.cells)
                    ):
                        inferred_cells = sentence.cells - sentence_comp.cells
                        if inferred_cells:
                            inferred_count = sentence.count - sentence_comp.count
                            inferred_sentence = Sentence(inferred_cells, inferred_count)
                            if (inferred_sentence not in self.knowledge and
                                inferred_sentence not in inferred_sentences
                            ):
                                inferred_sentences.append(inferred_sentence)
            if not inferred_sentences:
                break
            self.knowledge.extend(inferred_sentences)
            check_sentences = inferred_sentences

    def markMine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.markMine(cell)

    def markSafe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.markSafe(cell)

    def addKnowledge(self, cell, count):
        """
        Called when a safe cell on the Minesweeper board was revealed.
        Updates internal state and knowledge.
        cell: coordinates of the safe cell
        count: number of that cell (number of neighboring mines)
        """
        # Add cell to moves made, safe cells and add a new Sentence
        self.moves_made.add(cell)
        # A cell is not yet in self.safes if an unsafe (random) move was made.
        if cell not in self.safes:
            self.markSafe(cell)
        self.addSentence(cell, count)
        while True:
            self.inferSentences()
            # Query all Sentences for known safes and known mines.
            # Sentences that return something will be empty after updating them and will be removed.
            known_safes = set()
            known_mines = set()
            for sentence in self.knowledge:
                known_safes.update(sentence.knownSafes())
                known_mines.update(sentence.knownMines())
            if not known_safes and not known_mines:
                break
            self.safes.update(known_safes)
            self.mines.update(known_mines)
            # update Sentences with new safes and mines.
            for known_safe in known_safes:
                self.markSafe(known_safe)
            for known_mine in known_mines:
                self.markMine(known_mine)
            # remove empty sentences from knowledge
            empty_sentences = [sentence for sentence in self.knowledge if sentence.isEmpty()]
            self.knowledge = [sentence for sentence in self.knowledge if sentence not in empty_sentences]

    def makeSafeMove(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        Use a cell in self.safes that is not yet in self.moves_made.
        Return None if no safe move is found.
        """
        safe_moves = self.safes - self.moves_made
        if safe_moves:
            return next(iter(safe_moves))
        return None

    def makeRandomMove(self):
        """
        Returns a move to make on the Minesweeper board.
        This function is only called if makeSafeMove() returned None.
        Choose a cell that has not already been chosen and is not known to be a mine.
        """
        available_moves = set()
        for i in range(self.height):
            for j in range(self.width):
                move = (i, j)
                if move not in self.moves_made and move not in self.mines:
                    available_moves.add(move)
        if available_moves:
            return next(iter(available_moves))
        return None
