import math
import random
import time

# The game Nim begins with some number of piles, each with some number of objects.
# Players take turns: on a player’s turn, the player removes a number of objects from any one non-empty pile.
# Whoever removes the last object loses.

# The Reinforcement learning algorithm Q-learning will be trained on the game.
# The reward values are -1 for an action that looses the game, 1 for an action that wins the game and
# a value in between -1 and 1 (float) for an action that continues the game (consisting of current reward
# that is always 0 and future expected rewards).
# Function Q(s, a) outputs an estimate of the reward value (Q-value) of taking action a in state s.
# The state is represented by a list of numbers where each number is the current number of objects in a pile.
# An action is represented by the tuple (i, j), taking j objects from pile i.

# All Q-values are 0 initially. A Q-value exists for every (state, action) pair.
# Only when the game is over the Q-value of the current and previous (state, action) gets updated.
# When in the next game a state next to this previous state is reached, the Q-value of this state
# also gets updated, because it sees the future rewards from the state (used in the Q-learning formula).
# In this way the Q-value update propagates back to the start state.

class Nim():

    def __init__(self, initial=[1, 3, 5, 7]):
        """
        Initialize game board.
        Each game board has
            - `piles`: a list of how many elements remain in each pile
            - `player`: 0 or 1 to indicate which player's turn
            - `winner`: None, 0, or 1 to indicate who the winner is
        """
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def availableActions(cls, piles):
        """
        Nim.availableActions(piles) takes a `piles` list as input
        and returns all of the available actions `(i, j)` in that state.
        Action `(i, j)` represents the action of removing `j` items
        from pile `i` (where piles are 0-indexed).
        """
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @classmethod
    def otherPlayer(cls, player):
        """
        Nim.otherPlayer(player) returns the player that is not `player`.
        Assumes `player` is either 0 or 1.
        """
        return 0 if player == 1 else 1

    def switchPlayer(self):
        """
        Switch the current player to the other player.
        """
        self.player = Nim.otherPlayer(self.player)

    def move(self, action):
        """
        Make the move `action` for the current player. `action` is a tuple `(i, j)`.
        """
        pile, count = action

        # Check for errors
        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        # Update pile
        self.piles[pile] -= count
        self.switchPlayer()

        # Check for a winner
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class NimAI():

    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize the AI with an empty Q-learning dictionary,
        an alpha (learning) rate, and an epsilon (exploration) rate.
        The Q-learning dictionary maps `(state, action)` pairs to a Q-value
        (the reward value of function Q(s, a)).
        - `state` is a tuple of remaining piles, e.g. (1, 1, 4, 4).
        (`state` must be converted from a list to a tuple to be able to use it as a dictionary key)
        - `action` is a tuple `(i, j)` for an action.
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        """
        Updates Q-learning model, given an old state, an action taken in that state,
        a new resulting state, and the reward received from taking that action.
        """
        old = self.getQValue(old_state, action)
        best_future = self.bestFutureReward(new_state)
        self.updateQValue(old_state, action, old, reward, best_future)

    def getQValue(self, state, action):
        """
        Returns the Q-value for the state `state` and the action `action`.
        Returns 0 if no Q-value exists yet in `self.q`.
        """
        return self.q.get((tuple(state), action), 0)

    def updateQValue(self, state, action, old_q, reward, future_rewards):
        """
        Update the Q-value for the state `state` and the action `action` given the
        previous Q-value `old_q`, a current reward `reward`, and an estimate of
        future rewards `future_rewards`.
        The Q-learning formula is:
        Q(s, a) <- old_value_estimate + alpha * (new_value_estimate - old_value_estimate)
                <- Q(s, a) + alpha * (current_reward + best_future_reward - Q(s, a))
        """
        self.q[(tuple(state), action)] = old_q + self.alpha * (reward + future_rewards - old_q)

    def bestFutureReward(self, state):
        """
        Given a state `state`, considers all possible `(state, action)` pairs available
        in that state and returns the highest of their Q-values.
        Returns 0 If there are no available actions in `state`.
        """
        available_actions = Nim.availableActions(state)
        if not available_actions:
            return 0

        best_q = -math.inf
        for action in available_actions:
            best_q = max(best_q, self.getQValue(state, action))
        return best_q

    def chooseAction(self, state, use_epsilon=True):
        """
        Given a state `state`, return an action `(i, j)` to take.
        If `use_epsilon` is `False`, the best action available in the state is returned (greedy).
        If `use_epsilon` is `True`, a random available action with probability `self.epsilon`,
        and otherwise the best action available is chosen (epsilon-greedy).
        """
        epsilon = self.epsilon if use_epsilon else 0
        choose_random = True if random.random() < epsilon else False
        # assume there is always an available action
        available_actions = Nim.availableActions(state)
        if choose_random:
            return random.choice(list(available_actions))
        
        best_q = -math.inf
        best_action = None
        for action in available_actions:
            q_val = self.getQValue(state, action)
            if q_val > best_q:
                best_q = q_val
                best_action = action
        return best_action


def train(n):
    """
    Train an AI by playing `n` games against itself.
    Both players use the same AI and the AI is trained from the experiences of both players.
    """

    ai = NimAI()

    print(f"Play {n} training games")
    for _ in range(n):        
        game = Nim()

        # Keep track of last move made by either player
        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        # Game loop
        while True:

            # Keep track of current state and action
            state = game.piles.copy()
            action = ai.chooseAction(game.piles)

            # Keep track of last state and action
            last[game.player]["state"] = state
            last[game.player]["action"] = action

            # Make move and switch players
            game.move(action)
            new_state = game.piles.copy()

            # When game is over, update Q values with rewards
            if game.winner is not None:
                # The game is over when a player just made a move that lost him the game.
                # The move from the previous player was therefore game winning.
                # Both events are used to update the AI.
                # new_state is [0, 0, 0, 0] here and its used to update the AI, because
                # future rewards should not be considered in the Q-learning formula.
                ai.update(state, action, new_state, -1)
                ai.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    1
                )
                break

            # If game is continuing, no rewards yet
            elif last[game.player]["state"] is not None:
                ai.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    0
                )

    print("Done training")

    # Return the trained AI
    return ai


def play(ai, human_player=None):
    """
    Play human game against the AI.
    `human_player` can be set to 0 or 1 to specify whether
    human player moves first or second.
    """

    # If no player order set, choose human's order randomly
    if human_player is None:
        human_player = random.randint(0, 1)

    # Create new game
    game = Nim()

    # Game loop
    while True:

        # Print contents of piles
        print()
        print("Piles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")
        print()

        # Compute available actions
        available_actions = Nim.availableActions(game.piles)
        time.sleep(1)

        # Let human make a move
        if game.player == human_player:
            print("Your Turn")
            while True:
                pile = int(input("Choose Pile: "))
                count = int(input("Choose Count: "))
                if (pile, count) in available_actions:
                    break
                print("Invalid move, try again.")

        # Have AI make a move
        else:
            print("AI's Turn")
            pile, count = ai.chooseAction(game.piles, use_epsilon=False)
            print(f"AI chose to take {count} from pile {pile}.")

        # Make move
        game.move((pile, count))

        # Check for winner
        if game.winner is not None:
            print()
            print("GAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return
