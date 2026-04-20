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

        # initialisation de la table de transposition
        self.transposition_table = {}

    def act(self, state, remaining_time):

        # vide la table de transposition a chaque tour.
        # évite de dépasser la mémoire
        self.transposition_table.clear()

        # On récupère la profondeur max 
        depth_limit = adapt_depth(state, remaining_time)
        
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        
        # boucle sur tous les coups possibles
        actions = self.order_moves(state, True)
        for action in actions:
            next_state = state.copy()
            self.game.apply(next_state, action)

            if self.game.is_terminal(next_state):
                return action
            
            # On lance l'algorithme alpha-beta pour chaque coup
            score = self.alphabeta(next_state, depth_limit - 1, alpha, beta, False)
            
            # si un coup a une meilleur évaluation que best_move, on remplace best_move par ce coup
            if score > best_score:
                best_score = score
                best_move = action
            
            alpha = max(alpha, score) # on met a jour la borne si on a un meilleur coup
            
        return best_move

    # donne une clé unique pour un état
    def get_transposition_key(self, state):
        return (
            str(state.board), 
            self.game.to_move(state),
            state.totem_O,
            state.totem_X
        )

    """algorithme alpha-beta"""
    def alphabeta(self, state, depth, alpha, beta, maximizingPlayer):

        # calcule la clé du dictionnaire de l'état
        state_key = self.get_transposition_key(state)
        table_entry = self.transposition_table.get(state_key)

        # vérif si l'état est dans le dictionnaire et que la profondeur est sufisante
        if table_entry is not None and table_entry['depth'] >= depth:
            
            if table_entry['flag'] == 'EXACT':
                return table_entry['value']
            elif table_entry['flag'] == 'LOWERBOUND':
                alpha = max(alpha, table_entry['value'])
            elif table_entry['flag'] == 'UPPERBOUND':
                beta = min(beta, table_entry['value'])

            # pruning 
            if alpha >= beta:
                return table_entry['value']

        # condition d'arret
        if depth == 0 or self.game.is_terminal(state):
            if self.game.is_terminal(state):
                return self.game.utility(state, self.player) * 100000 # si jamais on gagne ou perds il faut un énorme poids
            return self.evaluate(state, self.player) 
        
        # On sauvegarde les bornes originales pour déterminer le flag à la fin
        original_alpha = alpha
        original_beta = beta

        # c'est à nous de jouer
        if maximizingPlayer:
            best_val = float('-inf')
            for action in self.order_moves(state, maximizingPlayer):
                # on joue le coup
                next_state = state.copy()
                self.game.apply(next_state, action)
                
                eval = self.alphabeta(next_state, depth - 1, alpha, beta, False)
                
                best_val = max(best_val, eval)
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break 

        # c'est à l'adversaire de jouer 
        else:
            best_val = float('inf')
            for action in self.order_moves(state, maximizingPlayer):
                next_state = state.copy()
                self.game.apply(next_state, action)
                
                eval = self.alphabeta(next_state, depth - 1, alpha, beta, True)
                
                best_val = min(best_val, eval)
                beta = min(beta, best_val)
                if beta <= alpha:
                    break 
        
        # on met dans la table ce qu'on vient de calculer
        flag = 'EXACT'
        if best_val <= original_alpha:
            flag = 'UPPERBOUND'
        elif best_val >= original_beta:
            flag = 'LOWERBOUND'

        self.transposition_table[state_key] = {
            'depth': depth,
            'value': best_val,
            'flag': flag
        }

        return best_val
    
        
        
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
        center_multiplier = max(0.0, (25.0 - plays) / 25.0) # voir si 15 est un bon nombre ?? 
                                                            # ca ne marche pas avec 15, mais ca marche mieux avec des valuers plus élevées. pourquoi ????

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
            # pour les windows de symboles
            is_my_turn = (current_to_move == player)
            for symbol in ['X', 'O']:
                score += score_symbol_window(window, symbol, is_my_turn)
        return score
    
    # ordonne les actions possibles en mettant les plus 
    # probables de conduire a un plateau favorable en premier
    def order_moves(self, state, maximizingPlayer):

        # On prépare les deux listes
        priority_moves = []
        others = []

        # liste les positions occupées par des jetons
        occupied = set() # plus rapide que une liste
        for r in range(6):
            for c in range(6):
                if state.board[r][c] is not None:
                    occupied.add((r, c))

        # 4 positions adjacentes a une case
        adjacent = [(-1, 0), (0, -1), (0, 1), (1, 0)]

        for action in list(self.game.actions(state)):

            # action[2] = tuple position de la piece posée
            r, c = action[2][0], action[2][1]
            
            is_near = False
            for dr, dc in adjacent:
                nr = r + dr 
                nc = c + dc
                if (nr, nc) in occupied:
                    is_near = True
                    break
            
            if is_near:
                priority_moves.append(action)
            else:
                others.append(action)
            
        # collage des 2 listes 
        return priority_moves + others


##################################################################
#################### fonctions auxiliaires #######################
##################################################################

def adapt_depth(state, remaining_time):
    "Ajustement dynamique mais avec des valeurs arbitraires pour l'instant, à améliorer"

    played = number_of_plays(state)
    print("adapt depth : ", played, remaining_time)

    # si presque plus de temps, on doit pas perdre de temps
    if remaining_time < 10:
        return 3

    # assez tôt, beaucoup de possibilités, on doit pas perdre de temps à explorer trop profondément
    if played < 5:
        print("early game")
        return 5

    # une fois plus de pièces posées, moins de possibilités. On peut rechercher plus en profondeur
    if played < 24:
        return 5

    # vers la fin : peu de possibilités, on peut se permettre de rechercher plus en profondeur pour trouver le meilleur coup
    if played > 30 :
        return 7

    return 5


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
"""
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
"""
def score_symbol_window(window, symbol, is_my_turn):
    count = 0
    for cell in window:
        if cell is not None:
            if cell[0] == symbol:
                count += 1
            else:
                # La ligne est bloquée par l'autre symbole (ex: un O bloque les X)
                # Impossible de faire 4, la fenêtre vaut 0 !
                return 0 
                
    if count == 3:
        # MENACE OU OPPORTUNITÉ IMMINENTE
        # Si c'est à moi de jouer, c'est une victoire (+20)
        # Si c'est à l'adversaire, je vais perdre (-20)
        return 100 if is_my_turn else -100
        
    elif count == 2:
        # DÉVELOPPEMENT NORMAL
        # On donne un petit bonus FIXE ET POSITIF (+1). 
        # Ainsi, l'IA ne fuira jamais une ligne de 2, même si c'est au tour de l'adversaire.
        # Cela l'encourage à construire et à rester au cœur du jeu !
        return 2
        
    return 0