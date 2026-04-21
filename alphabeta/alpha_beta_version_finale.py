
from agent import Agent
from oxono import Game
import time

class TimeoutException(Exception):
    pass

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
        # gestion du temps
        self.start_time = time.time()
        self.time_limit = time_function(state, remaining_time)
        
        # nétoyage de la table de transposition pour ne pas saturer la mémoire 
        self.transposition_table.clear()
        # garde en mémoire le meilleur mouvement de la profondeur suivante
        best_move_overall = None
        
        try:
            for depth in range(2, 20, 2): 
                alpha = float('-inf')
                beta = float('inf')
                best_score = float('-inf')
                best_move_for_this_depth = None
                
                # On trie les actions à la racine
                actions = self.order_moves(state, True)

                # on essaye en premier le meilleur coup trouvé de l'itération d'avant
                if best_move_overall in actions:
                    actions.remove(best_move_overall)
                    actions.insert(0, best_move_overall)

                for action in actions:
                    # Vérification du temps
                    if time.time() - self.start_time > self.time_limit:
                        raise TimeoutException()

                    next_state = state.copy()
                    self.game.apply(next_state, action)
                    
                    if self.game.is_terminal(next_state) and self.game.utility(next_state, self.player) == 1:
                        return action # Victoire immédiate
                    
                    # On lance la recherche
                    score = self.alphabeta(next_state, depth - 1, alpha, beta, False)
                    
                    if score > best_score:
                        best_score = score
                        best_move_for_this_depth = action
                    
                    alpha = max(alpha, score)
                
                best_move_overall = best_move_for_this_depth

        except TimeoutException:
            print(f" tour {number_of_plays(state)} : profondeur {depth}")
            pass # on sort de la boucle
            
        return best_move_overall
    
    # donne une clé unique pour un état
    def get_transposition_key(self, state):
        board_tuple = tuple(tuple(row) for row in state.board)
        return (
            board_tuple, 
            self.game.to_move(state),
            state.totem_O,
            state.totem_X
        )

    """algorithme alpha-beta"""
    def alphabeta(self, state, depth, alpha, beta, maximizingPlayer):

        # verification du temps
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutException()

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
            for action in self.game.actions(state):
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
            for action in self.game.actions(state):
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
        center_multiplier = max(0.0, (10.0 - plays) / 10.0) # voir si 15 est un bon nombre ?? 
                                                            # ca ne marche pas avec 15, mais ca marche mieux avec des valuers plus élevées. pourquoi ????

        for r in range(6):
            for c in range(6):
                cell = board[r][c]
                if cell is not None:
                    # cell[1] contient l'ID du joueur (0 ou 1)
                    if cell[1] == player:
                        score += POSITION_WEIGHTS[r][c] * center_multiplier 
                    else:
                        score -= POSITION_WEIGHTS[r][c] * center_multiplier 

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
            mult = 1
            if (current_to_move != player):
                mult = -1
            
            for symbol in ['X', 'O']:
                score += mult*score_symbol_window(window, symbol)
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
            
        # collage des 2 listes print(f"Tour {number_of_plays(state)} : Timeout atteint ! Profondeur max validée : {depth - 1}")
        return priority_moves + others


##################################################################
#################### fonctions auxiliaires #######################
##################################################################

# gère la répartition du temps
def time_function( state, remening_time):
        round_number = number_of_plays(state)
        if (round_number < 4):
            return 5
        if (round_number < 8):
            return 20
        return remening_time * 0.05

def number_of_plays(state):
    return sum(1 for row in state.board for cell in row if cell is not None)

# Attribue un score à une fenêtre selon le nombre de pieces de la bonne (ou mauvaise) couleur présente.
def score_color_window(window, color_player, sign):
    # Poids qui privilegie les lignes de 3
    weights = {3: 10, 2: 1}
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

    weights = {3: 10, 2: 1}
    count = 0
    
    for cell in window:
        if cell is not None :
            if cell[0] == symbol:
                count += 1
            else:
                return 0 # La ligne est bloquée par l'adversaire 

    return weights.get(count, 0)  # le 2eme argument est la si jamais get ne trouve pas la valeur count dans le dico
