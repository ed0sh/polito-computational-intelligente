# Tic Tac Toe with Reinforcement Learning
### Collaboration with Giovanni Bordero s313010 [@Giobordi](https://github.com/GioBordi)

The algorithm is based on the Q-learning, which is a model-free reinforcement learning algorithm. \
The agent learns from playing a number of games equal to the variable `TRAINING_EPOCHS`, against a random player. 
The reward is +2 if the agent wins, -5 if the agent loses and -1 if the game ends in a draw. These values have been 
chosen after some testing, tuning them to get the best results.

The agent exploits the symmetry of the game to learn only a reduced set of states, avoiding learning equivalent states 
multiple times. This technique helps the agent to learn faster and get better results (obviously the learning 
space is reduced).

The game-state is represented in a particular form, the Tic-Tac-Toe board is not treated as a linear vector of indexes.
Instead, it follows a spiral in order to easily manage the rotations and the symmetries.


### Code details

- The `class TicTacToe` is the environment in which the agent plays, it contains the methods to play a game and those 
to check if the game is over or not.

- There are the symmetry function `rotate_90_right` and `rotate_state_90_right` which are used to exploit the symmetry 
of the game.

- The `class ReinforcedPlayer` is the agent, it contains the Q table and the methods to update it and to choose the 
next action. The parameter *epsilon* helps to weight the exploration-exploitation tradeoff, it is set to 0.01.

- There are 2 function `save_model` and `load_model` to save and load the ReinforcedPlayer object, to avoid to train 
the agent every time.

