from collections import defaultdict
from copy import deepcopy

import random

from game import Player, Move, Game
from TrainingGame import TrainingGame

import symmetryUtils
import utils


class SymmetryMinMaxPlayer(Player):
    def __init__(self) -> None:
        super().__init__()
        self.max_depth = 2
        self.visited_states = defaultdict(int)

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        # Make a copy of the game to simulate moves
        training_game = TrainingGame(game.get_board(), game.get_current_player())

        # Recursively find the best move
        best_move_dict = self.minimax(training_game, 0, float('-inf'), float('inf'), True)

        return best_move_dict['move']

    def minimax(self, training_game: 'TrainingGame', depth: int, alpha: float, beta: float, is_maximizing: bool) -> dict:
        """Minimax algorithm"""

        # Get the current player index
        current_player = training_game.get_current_player() if is_maximizing \
            else abs(1 - training_game.get_current_player())

        # Evaluate terminal states
        if training_game.check_winner() == training_game.current_player_idx:
            return {'score': 10}
        elif training_game.check_winner() != training_game.current_player_idx and training_game.check_winner() != -1:
            return {'score': -10}
        elif depth >= self.max_depth:
            return {'score': training_game.get_intermediate_final_state_value(current_player)}

        scores = []

        # Get the possible moves form the current state and shuffle to avoid taking always the same moves
        available_moves_list = training_game.available_moves(current_player)
        random.shuffle(available_moves_list)

        if is_maximizing:
            for move in available_moves_list:
                move_dict = {
                    'move': move
                }

                # Simulate the move on the board
                game_to_simulate = deepcopy(training_game)
                game_to_simulate.move(
                    move[0],
                    move[1],
                    current_player
                )

                # Convert the state in a hashable way
                state = (utils.from_board_to_state_element(game_to_simulate.get_board()), move, current_player)

                # If the state hasn't been explored yet, go a step deeper in the recursion. Otherwise, take its score
                if self.get_visited_state_value(state) == 0:
                    result = self.minimax(game_to_simulate, depth + 1, alpha, beta, False)
                    score = result['score']
                    self.visited_states[state] = score
                else:
                    score = self.visited_states[state]

                move_dict['score'] = score
                scores.append(move_dict)

                # Perform alpha-beta pruning
                alpha = max(alpha, score)
                if beta <= alpha:
                    break

            # Take the best possible move from the explored ones (the one that maximizes the score)
            best_move = None
            best_score = float('-inf')
            for score in scores:
                if score['score'] > best_score:
                    best_score = score['score']
                    best_move = score
        else:
            for move in available_moves_list:
                move_dict = {
                    'move': move
                }

                # Simulate the move on the board
                game_to_simulate = deepcopy(training_game)
                game_to_simulate.move(
                    move[0],
                    move[1],
                    current_player
                )

                # Convert the state in a hashable way
                state = (utils.from_board_to_state_element(game_to_simulate.get_board()), move, current_player)

                # If the state hasn't been explored yet, go a step deeper in the recursion. Otherwise, take its score
                if self.get_visited_state_value(state) == 0:
                    result = self.minimax(game_to_simulate, depth + 1, alpha, beta, True)
                    score = result['score']
                    self.visited_states[state] = score
                else:
                    score = self.visited_states[state]

                move_dict['score'] = score
                scores.append(move_dict)

                # Perform alpha-beta pruning
                beta = min(beta, score)
                if beta <= alpha:
                    break

            # Take the best possible move from the explored ones (the one that minimizes the score)
            best_move = None
            best_score = float('inf')
            for score in scores:
                if score['score'] < best_score:
                    best_score = score['score']
                    best_move = score

        return best_move

    def get_visited_state_value(self, state: tuple) -> float:
        """return the Q value of the state"""

        policy = state
        # Rotate
        policy_90 = symmetryUtils.rotate_state_90_right(policy)
        policy_180 = symmetryUtils.rotate_state_180_right(policy)
        policy_270 = symmetryUtils.rotate_state_270_right(policy)
        # Mirror
        horizontal_mirror_policy = symmetryUtils.horizontal_mirror_state(policy)
        vertical_mirror_policy = symmetryUtils.vertical_mirror_state(policy)

        # Verify if the state has already been explored. If not, create an entry for it set to 0
        if policy not in self.visited_states \
                and policy_90 not in self.visited_states \
                and policy_180 not in self.visited_states \
                and policy_270 not in self.visited_states \
                and horizontal_mirror_policy not in self.visited_states \
                and vertical_mirror_policy not in self.visited_states:
            self.visited_states[policy] = 0

        # Return the value for the visited state
        if policy_90 in self.visited_states:
            return self.visited_states[policy_90]
        elif policy_180 in self.visited_states:
            return self.visited_states[policy_180]
        elif policy_270 in self.visited_states:
            return self.visited_states[policy_270]
        elif horizontal_mirror_policy in self.visited_states:
            return self.visited_states[horizontal_mirror_policy]
        elif vertical_mirror_policy in self.visited_states:
            return self.visited_states[vertical_mirror_policy]
        else:
            return self.visited_states[policy]
