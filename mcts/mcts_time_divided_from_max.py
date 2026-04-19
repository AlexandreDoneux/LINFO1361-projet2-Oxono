import math
import random
import time

from agent import Agent
from oxono import Game



# Time budget reserved per move (seconds).
#TODO -> other fixed times
#TODO -> dynamic time allocation
TIME_PER_MOVE = 1.0
# how to get the max time allowed? -> through act() ?



class MCTSNode:
    """
    Node in the MCTS search tree. Represents a gme state and tracks all related information needed for MCTS (visits, wins).

    Attributes
    ----------
    state        : the game state at this node
    parent       : the parent MCTSNode (None for the root)
    action       : the action that led from parent to this node (None for root)
    player       : the player who JUST MOVED to reach this state
                   (used to attribute wins correctly during backpropagation)
    children     : list of expanded child MCTSNodes
    untried_actions : actions not yet expanded into children
    visits       : number of times this node has been visited
    wins         : number of simulations won by `player` through this node
    """

    def __init__(self, state, parent=None, action=None, player=None):
        self.state           = state
        self.parent          = parent # -> to backpropagate results up the tree
        self.action          = action
        self.player          = player  # player who just moved #TODO-> is it neeeded ? -> can we improve to not have it ?

        self.children        = []
        self.untried_actions = list(Game.actions(state)) # ->move here to not call Game.actions(state) for each expansion

        # visits and wins are initialized to 0 and updated during backpropagation
        self.visits          = 0
        self.wins            = 0

    def is_fully_expanded(self):
        """Return True when no more legal actions are left from this node."""
        return len(self.untried_actions) == 0

    def is_terminal(self):
        """Return True if this node's state is a terminal game state (win, loss, draw)."""
        return Game.is_terminal(self.state)

    def ucb1(self):
        """
        Compute the UCB1 score of the node.

        UCB1 = wins/visits + C * sqrt(ln(parent.visits) / visits)
        #TODO -> for now C = sqrt(2), better value ?

        A node with many wins and few visits scores high.
        A node that has never been visited scores +∞ (always explored first).

        Returns
        -------
        float : UCB1 score
        """
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration  = math.sqrt(2) * math.sqrt(math.log(self.parent.visits) / self.visits) # confidence measure set to sqrt(2) (comon choice
        return exploitation + exploration
    # CHECK IF WORKS CORRECTLY

    def best_child(self):
        """
        Returns the child with highest UCB1 score.
        """
        return max(self.children, key=lambda c: c.ucb1())

    def best_action_child(self):
        """
        Returns the child with the most visits. Used at the end to pick the final move (most visited = most reliable).
        """
        return max(self.children, key=lambda c: c.visits)

    def expand(self):
        """
        Expansion phase: pick one untried action, create its child node, add it to children, and return it.
        """
        action = self.untried_actions.pop()   # pick an action still available
        next_state = self.state.copy() # copy to not change the current node's state when we apply the action
        Game.apply(next_state, action)

        child = MCTSNode(
            state  = next_state,
            parent = self,
            action = action,
            player = Game.to_move(self.state)     # the player who just moved
        )
        self.children.append(child)
        return child



class MCTSAgent(Agent):
    """
    Agent using Monte Carlo Tree Search (MCTS). UCT version # -> can we say version ? is UCT a version of MCTS ? or is it just a specific way to compute the UCB1 score ?
    Each call to act() runs as many MCTS iterations as possible within the allowed time, then returns the action of the
    most-visited child of the root.

    #TODO-> add time variation for better search (like for minimax/alphabeta agents)

    4 phases for each iteration :
        1. Selection   : walk the tree using UCB1 until finding a a non-fully-expanded or terminal node.
        2. Expansion   : if the node is not terminal, add one new child (without choosing for the moment)
        3. Simulation  : from the new child, play random moves until the game ends (rollout).
        4. Backprop    : propagate the result back up to the root, updating visits and wins at every node above
    """

    def __init__(self, player):
        super().__init__(player)

    def act(self, state, remaining_time):
        """
        Run MCTS for TIME_PER_MOVE seconds, returns the best action found.

        Parameters
        ----------
        state          : current game state
        remaining_time : total seconds left on our clock for the whole game

        Returns
        -------
        tuple : the action of the most-visited child of the root
        """
        # Build the root node for this turn
        root = MCTSNode(state=state.copy(), parent=None, action=None, player=None)

        # Time allowed for this move
        # limit somehow to avoid timeout -> laer when testing with more time per move
        deadline = time.time() + TIME_PER_MOVE

        # Run iterations until time runs out
        while time.time() < deadline:
            self.iterate(root)

        # Return the action of the most-visited child
        return root.best_action_child().action


    def iterate(self, root):
        """
        Run one full MCTS iteration: Selection → Expansion → Simulation → Backprop.
        """

        # Slection
        # Walk down the tree choosing the child with the best UCB1, until node not fully expanded or is terminal.
        node = root
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.best_child()

        # Expansion
        # If the node is not terminal and has remaining actions possible, expand one node
        if not node.is_terminal():
            node = node.expand()

        # Simulation
        # From that node, play random moves until the game ends.
        result = self.simulate(node.state)

        # Backpropagation
        # Walk back up to the root, updating visits and wins at every node.
        self.backpropagate(node, result)



    def simulate(self, state):
        """
        Simulation phase: play random moves from `state` until the game ends.
        #TODO -> always random or can take heuristics ? -> check later for better version

        Parameters
        ----------
        state : the state to simulate from (will be modified in place)

        Returns
        -------
        int : utility for self.player at the terminal state (1, -1, or 0)
        #TODO-> better utility ? -> check later for better version, would need to change the static function from the Game class, is it permitted ?
        """
        simulation_state = state.copy()

        while not Game.is_terminal(simulation_state):
            action = random.choice(list(Game.actions(simulation_state)))
            Game.apply(simulation_state, action)

        # 0 if draw, -1 if loose, 1 if win
        return Game.utility(simulation_state, self.player)



    def backpropagate(self, node, result):
        """
        go from current node back to the root, incrementing visits at every node and incrementing wins when the
        node's player matches the winner from simulation.

        Parameters
        ----------
        node   : the node where simulation started
        result : utility for self.player (1 = win, -1 = loss, 0 = draw)
        """
        while node is not None:
            node.visits += 1

            # Adding wins only for the player who just moved at this node , works now
            if node.player is not None:
                if node.player == self.player and result == 1:
                    node.wins += 1
                elif node.player != self.player and result == -1:
                    node.wins += 1

            node = node.parent