from agent import Agent
from oxono import Game


# Depth limit for Alpha-Beta search.
# Alpha-Beta prunes des branches -> on peut se permettre d'explorer plus profondément que minimax
SEARCH_DEPTH = 4


class AlphaBetaAgent(Agent):
    """
    Oxono agent using the Alpha-Beta pruning algorithm. Based on the pseudo-code from the lectures.
    Available in "Artificial intelligence : a modern approach" by Russell and Norvig,
    fourth edition, chapter 6, page 200.

    Very similar to Minimax, it only adds the principle of prunng to avoid searching branches that won't be chosen by rational players.


    It tracks two values :
        alpha : the best score MAX is already guaranteed (starts at -∞)
        beta  : the best score MIN is already guaranteed (starts at +∞)

        In MAX-VALUE: if v >= beta  → stop (MIN would never allow this branch)
        In MIN-VALUE: if v <= alpha → stop (MAX would never choose this branch)
    """

    def __init__(self, player):
        super().__init__(player)


    def act(self, state, remaining_time):
        """
        from the pseudo-code:
            player <- game.TO-MOVE(state)
            value, move <- MAX-VALUE(game, state, -∞, +∞)
            return move

        Parameters
        ----------
        state          : current game state
        remaining_time : seconds left on our clock for the whole game

        Returns
        -------
        The best action found by Alpha-Beta at depth SEARCH_DEPTH.
        """

        depth = 0
        _, best_move = self.max_value(state, depth, alpha=float('-inf'), beta=float('+inf')) # alpha and beta initialosed
        return best_move


    def max_value(self, state, depth, alpha, beta):
        """
        from the pseudo-code:

            if game.IS-TERMINAL(state) then return game.UTILITY(state, player), null
            v, move <- −∞
            for each a in game.ACTIONS(state) do
                v2, a2 <- MIN-VALUE(game, game.RESULT(state, a), alpha, beta)
                if v2 > v then
                    v, move <- v2, a
                    alpha <- MAX(alpha, v)
                if v >= beta then return v, move   ← PRUNING: MIN will never allow this
            return v, move

        Parameters
        ----------
        state : current game state
        depth : current depth in the tree (incremented at each recursive call)
        alpha : best score MAX is already guaranteed on the path to the root
        beta  : best score MIN is already guaranteed on the path to the root

        Returns
        -------
        (score, action) : best score achievable and the move that leads to it.
                          action is None at terminal/leaf nodes.
        """
        if Game.is_terminal(state):
            return Game.utility(state, self.player), None

        if depth == SEARCH_DEPTH:
            return evaluate(state, self.player), None

        v = float('-inf')
        best_move = None

        for action in Game.actions(state):
            next_state = state.copy()
            Game.apply(next_state, action)

            v2, _ = self.min_value(next_state, depth + 1, alpha, beta)

            # Keep track of the best score
            if v2 > v:
                v = v2
                best_move = action

            # Update alpha
            alpha = max(alpha, v)

            # Puning
            if v >= beta:
                return v, best_move

        return v, best_move


    def min_value(self, state, depth, alpha, beta):
        """
        from the pseudo-code:

            if game.IS-TERMINAL(state) then return game.UTILITY(state, player), null
            v, move <- +∞
            for each a in game.ACTIONS(state) do
                v2, a2 <- MAX-VALUE(game, game.RESULT(state, a), alpha, beta)
                if v2 < v then
                    v, move <- v2, a
                    beta <- MIN(beta, v)
                if v <= alpha then return v, move   ← PRUNING: MAX will never choose this
            return v, move

        Parameters
        ----------
        state : current game state
        depth : current depth in the tree (incremented at each recursive call)
        alpha : best score MAX is already guaranteed on the path to the root
        beta  : best score MIN is already guaranteed on the path to the root

        Returns
        -------
        (score, action) : lowest score the opponent can force and the move
                          that achieves it. action is None at terminal/leaf nodes.
        """
        if Game.is_terminal(state):
            return Game.utility(state, self.player), None

        if depth == SEARCH_DEPTH:
            return evaluate(state, self.player), None

        v = float('+inf')
        best_move = None

        for action in Game.actions(state):
            next_state = state.copy()
            Game.apply(next_state, action)

            v2, _ = self.max_value(next_state, depth + 1, alpha, beta)

            # Keep track of the best score
            if v2 < v:
                v = v2
                best_move = action

            # Update beta
            beta = min(beta, v)

            # Pruning
            if v <= alpha:
                return v, best_move

        return v, best_move


def evaluate(state, player):
    return 1
# => need to have real heuristic. If we return a constant, the agent will not be able to distinguish between non-terminal states and will just choose the first action available.