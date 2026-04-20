from agent import Agent
from oxono import Game

# Depth limit for Alpha-Beta search.
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
        
        # initialisation de la liste de colones et lignes de 4
        self.game = Game()
        self.windows = []
        for r in range(6):
            for c in range(3):
                self.windows.append([(r, c + i) for i in range(4)])
        for c in range(6):
            for r in range(3):
                self.windows.append([(r + i, c) for i in range(4)])

    def act(self, state, remaining_time):

        # On récupère la profondeur max 
        depth_limit = adapt_depth(state, remaining_time)
        
        best_move = None
        max_val = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        # boucle sur tous les coups possibles
        for action in self.game.actions(state):
            next_state = state.copy()
            self.game.apply(next_state, action)
            
            # On lance l'algorithme alpha-beta pour chaque coup
            val = self.alphabeta(next_state, depth_limit - 1, alpha, beta, False)
            
            # si un coup a une meilleur évaluation que best_move, on remplace best_move par ce coup
            if val > max_val:
                max_val = val
                best_move = action
            
            alpha = max(alpha, val) # on met a jour la borne si on a un meilleur coup
            
        return best_move

    def alphabeta(self, state, depth, alpha, beta, maximizingPlayer):
        
        # condition d'arret
        if depth == 0 or self.game.is_terminal(state):
            if self.game.is_terminal(state):
                return self.game.utility(state, self.player) * 100000 # si jamais on gagne ou perds il faut un énorme poids
            # Ajout de self.player comme argument attendu
            return self.evaluate(state, self.player) 

        # c'est à nous de jouer
        if maximizingPlayer:
            maxEval = float('-inf')
            for action in self.game.actions(state):
                # on joue le coup
                next_state = state.copy()
                self.game.apply(next_state, action)
                
                eval = self.alphabeta(next_state, depth - 1, alpha, beta, False)
                
                maxEval = max(maxEval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break 
            return maxEval

        # c'est à l'adversaire de jouer 
        else:
            minEval = float('inf')
            for action in self.game.actions(state):
                next_state = state.copy()
                self.game.apply(next_state, action)
                
                eval = self.alphabeta(next_state, depth - 1, alpha, beta, True)
                
                minEval = min(minEval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break 
            return minEval
        
    # Estime la valeur d'un plateau non terminal.
    def evaluate(self, state, player):
        opponent = 1 - player
        score = 0
        current_to_move = self.game.to_move(state)
        board = state.board
        plays = number_of_plays(state)

        """partie controle du centre"""
        # au début, avoir des pieces au centre est plus avantageuses
        POSITION_WEIGHTS = [
            [1, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 2, 3, 3, 2, 1],
            [1, 2, 3, 3, 2, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 1, 1, 1, 1, 1]
        ]
        
        # fin de la partie --> avoir des lignes > placer les pieces au centre
        center_multiplier = max(0.0, (15.0 - plays) / 15.0) # voir si 15 est un bon nombre ??

        for r in range(6):
            for c in range(6):
                cell = board[r][c]
                if cell is not None:
                    # cell[1] contient l'ID du joueur (0 ou 1)
                    if cell[1] == player:
                        score += POSITION_WEIGHTS[r][c] * center_multiplier * 3.0
                    else:
                        score -= POSITION_WEIGHTS[r][c] * center_multiplier * 3.0

        """partie lignes de 4"""

        for window_coords in self.windows:
            # on transforme la liste de coordonnées en liste de cell 
            window = [board[r][c] for r, c in window_coords]

            # score_color_window attribue un score a une fenetre selon combien de pieces de la couleur sont dedans 
            val_p = score_color_window(window, player, +1)
            val_o = score_color_window(window, opponent, -1)

            # plus élevé pour le joueur qui va jouer au prochain tour
            mult_p = 2.0 if current_to_move == player else 1.0
            mult_o = 2.0 if current_to_move == opponent else 1.0
            score += val_p * mult_p
            score += val_o * mult_o

            # pour les windows de symboles
            for symbol in ['X', 'O']:
                symb_val = score_symbol_window(window, symbol)
                score += symb_val * (1 if current_to_move == player else -1)

        return score

def adapt_depth(state, remaining_time):
    "Ajustement dynamique mais avec des valeurs arbitraires pour l'instant, à améliorer"

    played = number_of_plays(state)
    print("adapt depth : ", played, remaining_time)

    # si presque plus de temps, on doit pas perdre de temps
    if remaining_time < 10:
        return 3

    # assez tôt, beaucoup de possibilités, on doit pas perdre de temps à explorer trop profondément
    if played < 10:
        print("early game")
        return 4

    # une fois plus de pièces posées, moins de possibilités. On peut rechercher plus en profondeur
    if played < 24:
        return 4

    # vers la fin : peu de possibilités, on peut se permettre de rechercher plus en profondeur pour trouver le meilleur coup
    if played > 30 :
        return 8

    return 6


def number_of_plays(state):
    return sum(1 for row in state.board for cell in row if cell is not None)

# Attribue un score à une fenêtre selon le nombre de pieces de la bonne (mauvaise) couleur présentse.
def score_color_window(window, color_player, sign):
    # Poids qui privilegie les lignes de 3
    weights = {3: 20, 2: 2}
    opponent = 1 - color_player

    my_count = 0
    empty_count = 0
    
    for cell in window:
        if cell is not None:
            if cell[1] == color_player:
                my_count += 1
            else:
                # la fenètre est bloquée par l'adversaire pour la couleur
                return 0
        else:
            empty_count += 1

    return sign * weights.get(my_count, 0)

# Attribue un score à une fenêtre selon le nombre de symboles (X ou O) présents.
def score_symbol_window(window, symbol):
    # Les symboles ont moins de poids car tout le monde peut gagner avec
    weights = {3: 10, 2: 1}
    count = 0
    
    for cell in window:
        if cell is not None and cell[0] == symbol:
            count += 1
            weights.get(count, 0)
    
    return weights.get(count, 0)  # le 2eme argument est la si jamais get ne trouve pas la valeur count dans le dico
