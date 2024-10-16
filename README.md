# monte carlo preflop equity charts

Welcome to this repository! See `main.ipynb`. 

Equity is calculated for each unique hero vs villain hand-pair by giving the "hero" player each possible pair of hole-cards where the higher card is a spade and the other card is either a spade or a club 
(equivalent without loss of generality or accuracy to any possible suited or offsuit hand since suits don't matter in Texas Hold Em), and randomly selecting seven cards from a full deck to make up the 
"villain" player's hole cards (the first two cards of the seven) and the board (cards three through seven) for each iteration. For each of the hero's possible sets of hole-cards, a showdown if simulated if 
they don't conflict with the seven cards dealt as the board and villain's hole cards and equity is calcuated for a hole-card chart chart by mapping hole-card-pairs with specific suits (ex: AhKd vs JhJc) to 
hole-card-pairs containing only relative suiting information (ex: AKo vs JJ) in a large dictionary. 

Results are visualized as hand matrices using matplotlib.
