"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None

def getInitialState():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

def getPlayer(board):
    """
    Returns player who has the next turn on a board.
    Player X is always starting.
    """
    count_x, count_o = 0, 0
    for row in board:
        count_x += row.count(X)
        count_o += row.count(O)
    if count_x > count_o:
        return O
    return X

def getActions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    for row_i in range(len(board)):
        for col_i in range(len(board[0])):
            if board[row_i][col_i] == EMPTY:
                actions.add((row_i, col_i))
    return actions

def getResult(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    result_board = deepcopy(board)
    player = getPlayer(board)
    result_board[action[0]][action[1]] = player
    return result_board

def getWinner(board):
    """
    Returns the winner of the game, if there is one.
    """
    players = [X, O]
    num_symbols_in_line = 3
    for player in players:
        # check rows
        for row in board:
            line_count = row.count(player)
            if line_count == num_symbols_in_line:
                return player
        
        # check columns
        for col_i in range(len(board[0])):
            line_count = 0
            for row_i in range(len(board)):
                if board[row_i][col_i] == player:
                    line_count += 1
            if line_count == num_symbols_in_line:
                return player
        
        # check vertical from top left to bottom right
        line_count = 0
        for vert_cell in range(len(board)):
            if board[vert_cell][vert_cell] == player:
                line_count += 1
        if line_count == num_symbols_in_line:
            return player
        
        # check vertical from top right to bottom left
        line_count = 0
        col_i = len(board) - 1
        for row_i in range(len(board)):
            if board[row_i][col_i] == player:
                line_count += 1
            col_i -= 1
        if line_count == num_symbols_in_line:
            return player

    return None

def isTerminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if getWinner(board):
        return True
    
    count_empty = 0
    for row in board:
        count_empty += row.count(EMPTY)
    if count_empty == 0:
        return True
    return False

def getUtility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    This function may only be called if terminal(board) is True.
    """
    winner = getWinner(board)
    if winner == X:
        return 1
    if winner == O:
        return -1
    return 0

# Implementation of Alpha-Beta Pruning:
# The function minimax additionally gives the current optimal score and the current player
# to the minValue/ maxValue functions.
# To get an initial value for the optimal score (other than +-inf), at least one subtree of actions has to be
# traversed fully (the subtree of all possible games after one action was chosen).
# For player X: if the minValue sees a value <= the current optimal value, the subtree does not need to be
#   considered any further, because it will only ever yield a value that is <= the current optimal value.
# For player O: if the maxValue sees a value >= the current optimal value, the subtree does not need to be
#   considered any further, because it will only ever yield a value that is >= the current optimal value.
# The efficiency of Alpha-Beta Pruning can be increased even further if the best and worst possible scores are known.
# For player X: if minValue finds a -1 (worst value) it will always pick it so the subtree does not need to be considered further
#   (helps when traversing the first subtree when the current optimal score is -inf).
#   if maxValue finds a 1 it will always pick it so the subtree does not need to be considered further.
# For player O: if maxValue finds a 1 (worst value) it will always pick it so the subtree does not need to be considered further
#   (helps when traversing the first subtree when the current optimal score is inf).
#   if minValue finds a -1 it will always pick it so the subtree does not need to be considered further.

def minValue(board, cur_optimal_val, player):
    """
    For a given state, return the minimum score that Player O can achieve
    considering Player X plays optimally.
    """
    # base case (leave recursion)
    if isTerminal(board):
        return getUtility(board)

    val = math.inf
    for action in getActions(board):
        val = min(val, maxValue(getResult(board, action), cur_optimal_val, player))        
        if (
            (player == X and (val <= cur_optimal_val or val == -1)) or
            (player == O and val == -1)
        ):
            break
    return val

def maxValue(board, cur_optimal_val, player):
    """
    For a given state, return the maximum score that Player X can achieve
    considering Player O plays optimally.
    """
    # base case (leave recursion)
    if isTerminal(board):
        return getUtility(board)

    val = -math.inf
    for action in getActions(board):
        val = max(val, minValue(getResult(board, action), cur_optimal_val, player))
        if (
            (player == O and (val >= cur_optimal_val or val == 1)) or
            (player == X and val == 1)
        ):
            break
    return val

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    optimal_action = None
    player = getPlayer(board)
    if player == X:
        # Player X tries to find the action that maximizes his score.
        # He must therefore look how player O tries to minimize the score after each available action.
        # From all available actions, the action with the highest score is picked.
        # When the board is empty, the score of all actions is 0.
        # i.e. no matter where the first symbol is placed, if both play optimally the game will end in a tie.
        optimal_val = -math.inf
        for action in getActions(board):
            val = minValue(getResult(board, action), optimal_val, player)
            if val > optimal_val:
                optimal_val = val
                optimal_action = action
                # The action to win the game is found, so don't consider other actions
                if val == 1:
                    break
    else:
        # Player O tries to find the action that minimizes his score.
        # He must therefore look how player X tries to maximize the score after each available action.
        # From all available actions, the action with the lowest score is picked.
        optimal_val = math.inf
        for action in getActions(board):
            val = maxValue(getResult(board, action), optimal_val, player)
            if val < optimal_val:
                optimal_val = val
                optimal_action = action
                # The action to win the game is found, so don't consider other actions
                if val == -1:
                    break

    return optimal_action
