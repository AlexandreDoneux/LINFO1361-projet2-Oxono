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
# - laisser un temps plus en moins long en fonction du temps restant, du nombre de pièces déjà passées et de la valeur de la fonction d'évaluation de l'état
# - optimiser le code python pour gagner du temps à l'exécution (ex: éviter les boucles for imbriquées, utiliser des structures de données plus efficaces, etc.)
# => gain de temps général et recherche o-plus éfficace
# - faire des recherches sur un nombre limité de coups en avance (ex: 2 ou 3) et utiliser une fonction d'évaluation pour estimer la valeur de l'état à la fin de la recherche.
# Cela permet de gagner du temps en évitant de rechercher jusqu'à la fin du jeu, tout en ayant une estimation de la valeur de l'état.
#    => à utiliser dès le début du projet pour éviter de perdre du temps à implémenter une recherche jusqu'à la fin du jeu, ce qui est très coûteux en temps de calcul.
# - améliorer les fonction d'évaluation pour qu'elles soient plus précises et prennent en compte plus de facteurs (ex: nombre de pièces alignées, nombre de pièces bloquées, etc.)
#       => pour le moment audune idée...
#
