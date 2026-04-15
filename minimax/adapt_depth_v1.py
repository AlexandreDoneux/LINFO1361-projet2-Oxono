from agent import Agent
from oxono import Game


# Depth limit for Minimax search.
SEARCH_DEPTH = 2 # -> peut être ajusté. Faire comparaisons entre différentes valeurs pour un compromis qualité - vitesse ?
# améliorations : ajustement dynamique en fonction du temps restant, du nombre de pièces déjà passées, de la valeur de la fonction d'évaluation, ...


class MinimaxAgent(Agent):
    """
    Oxono agent using the Minimax algorithm. Based on the pseudo-code from the lectures. Available in
    "Artificial intelligence : a modern approach" by Russell and Norvig, fourth edition, chapter 6, page 196.

    The algorithm explores the game tree up to SEARCH_DEPTH, then give scores to non-terminal states with a heuristic evaluation function.
    """

    def __init__(self, player):
        super().__init__(player)


    def act(self, state, remaining_time):
        """
        from the pseudo-code:
            player<- game.TO-MOVE(state) => needed ? -> No because we are always MAX
            value, move <- MAX-VALUE(game,state)
            return move

        Parameters
        ----------
        state          : current game state
        remaining_time : seconds left on our clock for the whole game -> need to be used

        Returns
        -------
        The best action found by Minimax at depth SEARCH_DEPTH.
        """
        # always start as MAX ? -> yes because we are the ones choosing the move
        depth = 0 # to track depth
        global SEARCH_DEPTH
        SEARCH_DEPTH = adapt_depth(state, remaining_time) # to adjust depth limit based on time remaining and game state
        _, best_move = self.max_value(state, depth)
        return best_move


    def max_value(self, state, depth):
        """
        from the pseudocode:

            if game.IS-TERMINAL(state) then return game.UTILITY(state, player), null
            v, move <- −∞
            for each a in game.ACTIONS(state) do
                v2, a2 <- MIN-VALUE(game, game.RESULT(state, a))
                if v2 > v then
                    v, move <- v2, a
            return v, move

        We add a depth limit: when depth reaches 0 on a non-terminal state,
        we call the heuristic instead of recursing further.

        Returns
        -------
        (score, action) : best score achievable and the move that leads to it.
                          action is None at terminal/leaf nodes.
        """


        # if terminal state: return true utility
        if Game.is_terminal(state):
            return Game.utility(state, self.player), None

        # if depth limit reached: estimate with heuristic
        if depth == SEARCH_DEPTH:
            return evaluate(state, self.player), None

        # else: look for the highest score among possible actions
        v = float('-inf')
        best_move = None

        for action in Game.actions(state):
            next_state = state.copy()
            Game.apply(next_state, action)

            # Recurse into MIN
            v2, _ = self.min_value(next_state, depth + 1)

            # Keep track of the best score
            if v2 > v:
                v = v2
                best_move = action

        return v, best_move


    def min_value(self, state, depth):
        """
        from the pseud-ocode:

            if game.IS-TERMINAL(state) then return game.UTILITY(state, player), null
            v, move <- +∞
            for each a in game.ACTIONS(state) do
                v2, a2 <- MAX-VALUE(game, game.RESULT(state, a))
                if v2 < v then
                    v, move <- v2, a
            return v, move

        Returns
        -------
        (score, action) : lowest score the opponent can force and the move
                          that achieves it. action is None at terminal/leaf nodes.
        """
        # if terminal state: return true utility
        if Game.is_terminal(state):
            return Game.utility(state, self.player), None

        # if depth limit reached: estimate with heuristic
        if depth == SEARCH_DEPTH:
            return evaluate(state, self.player), None

        # else
        v = float('+inf')
        best_move = None

        for action in Game.actions(state):
            next_state = state.copy()
            Game.apply(next_state, action)

            # Recurse into MAX
            v2, _ = self.max_value(next_state, depth + 1)

            # Keep track of the best score
            if v2 < v:
                v = v2
                best_move = action

        return v, best_move



def evaluate(state, player):
    # return constant ? ->  Il faut d'office quelque chose de basique mais pas une constante, sinon on perdrait du temps à
    # explorer le même nombre de noeuds à chaque fois, sans jamais pouvoir différencier les coups.
    return 1


def adapt_depth(state, remaining_time):
    "Ajustement dynamique mais avec des valeurs arbitraires pour l'instant, à améliorer"

    played = number_of_plays(state)
    print("adapt depth : ", played, remaining_time)

    # si presque plus de temps, on doit pas perdre de temps
    if remaining_time < 10:
        return 2

    # assez tôt, beaucoup de possibilités, on doit pas perdre de temps à explorer trop profondément
    if played < 10:
        print("early game")
        return 3

    # une fois plus de pièces posées, moins de possibilités. On peut rechercher plus en profondeur
    if played < 24:
        return 4

    # vers la fin : peu de possibilités, on peut se permettre de rechercher plus en profondeur pour trouver le meilleur coup
    if played > 30 :
        return 6

    return 5


def number_of_plays(state):

    return sum(1 for row in state.board for cell in row if cell is not None)
