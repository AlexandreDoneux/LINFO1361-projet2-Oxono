from agent import Agent
from oxono import Game
import random

class RandomAgent(Agent):
    def __init__(self, player):
        super().__init__(player)
    
    def act(self, state, remaining_time):
        actions = list(Game.actions(state))
        return random.choice(actions)





# idées d'heuristiques et techniques
# - éviter de placer des pièces d'un type (x ou o) trop loin de celles déjà placées. On doit en aligner 4 don on évite
# d'avoir un espace de 3 ou plus entre la pièce d'un type et les autres.
# - laisser un temps plus en moins long en fonction du temps restant, du nombre de pièces déjà passées et de la valeur de la fonction d'évaluation de l'état a
#
#
#
