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


def adapt_depth(state, remaining_time):
    "Ajustement dynamique mais avec des valeurs arbitraires pour l'instant, à améliorer"

    played = number_of_plays(state)

    # si presque plus de temps, on doit pas perdre de temps
    if remaining_time < 10:
        return 3

    # assez tôt, beaucoup de possibilités, on doit pas perdre de temps à explorer trop profondément
    if played < 10:
        return 4

    # une fois plus de pièces posées, moins de possibilités. On peut rechercher plus en profondeur
    if played < 24:
        return 5

    # vers la fin : peu de possibilités, on peut se permettre de rechercher plus en profondeur pour trouver le meilleur coup
    return 6


def number_of_plays(state):

    return sum(1 for row in state.board for cell in row if cell is not None)


def evaluate(state, player):
    opponent = 1 - player
    score = 0

    windows = get_windows(state.board)

    for window in windows:
        score += score_color_window(window, player, +1)
        score += score_color_window(window, opponent, -1)

    return score



# warning : but : aligner 4 pièces du même symbole où de la même couleur, mais aussi empêcher l'adversaire de le faire

# Quelques règles/idées que j'ai trouvé :

# Éviter les longues lignes de la couleur de l'adversaire : Vu que seul l'adversaire peut poser sa couleur faut éviter
# les longues lignes dans sa couleur. Si il y a des lignes de 3 en sa couleur qui peuvent encore être complétés
# => pas bon du tout, on doit diminuer fortement la valeur de cet état. Idem pour les lignes de deux à moindre mesure.
# On doit aussi prendre en compte les lignes pouvant être complétées par le centre (genre [Black] [vide] [Black] [Black] est très dangereux).
#
# À l'inverse privilégier les lignes de notre couleur (en suivant les mêmes règles). On augmentera le score de l'état dans ces cas là.

# Pour l'alignement de symboles nos pièces peuvent servir à l'adversaire et inversement. Donc l'évaluation de ce qui est
# bien ou non est plus compliqué. Je n'ai pas encore trouvé quelque chose d'intéressant pour ça.

# Les règles concernant un totem bloqué peut être intéressant vu qu'on limites les choix mais ça peut être à notre
# avantage comme à l'avantage de l'adversaire. Je pense que ça peut être intéressant d'y regarder.


def get_windows(board):
    """
    Return all horizontal and vertical slices of 4 consecutive cells on the board
    """
    windows = []

    # horizontal
    for r in range(6):
        for c in range(3):
            windows.append([board[r][c + i] for i in range(4)])
    # vertical
    for c in range(6):
        for r in range(3):
            windows.append([board[r + i][c] for i in range(4)])

    return windows


def score_color_window(window, color_player, sign):
    """
    Score window of 4 cells (does not matter if vertical or horizontal, they are passed as 4 values) for a player's color.
    Gives a score to a window that can be completed (no pieces of the opponent's color).
    """
    weights = {3: 500, 2: 10, 1: 1} # autres valeurs, 3 -> très bon ou trè dangereux, 1 -> sert à quelque chose de mettre une veleur ?

    opponent = 1 - color_player

    # Count pieces of each color in the window
    my_count = sum(1 for cell in window if cell is not None and cell[1] == color_player)
    opponent_count = sum(1 for cell in window if cell is not None and cell[1] == opponent)

    # Si un pièce de l'adversaire -> pas de score pour les 4 cases (on peut pas compléter la ligne)
    if opponent_count > 0:
        return 0

    return sign * weights.get(my_count,0) # sign multiplies the value by +1 if it's good for us, -1 if it's good for the opponent


# CRASH dès le début...