import random
from game import Game, Move, Player
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from copy import deepcopy

from MinMaxPlayer import SymmetryMinMaxPlayer
from utils import RandomPlayer, save_model
from TrainingGame import TrainingGame
import math
import multiprocessing as mp
import symmetryUtils
import utils


class MultiSymmetryAgent(Player):
    def __init__(self, training_match: int = 10_000, reward_win: int = 1, reward_loss: int = -3) -> None:
        super().__init__()
        self.current_match = None
        self.Q = defaultdict(float)
        self.match_moves = []
        self.train = True
        self.training_match = training_match
        self.reward_win = reward_win
        self.reward_loss = reward_loss
        self.default_player = 0

        self.epsilon = 0.3  ## exploration rate
        self.learning_rate = 0.3
        self.discount_factor = 0.9
        self.decrease_epsilon_epoch = training_match // 10
        self.decrease_epsilon = 0.02

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        # the board from the game object is not hashable, so we need to convert it to a tuple
        current_board = utils.from_board_to_state_element(game.get_board())

        # we train the model only with one player, the other one can be the flipped version
        # also to save memory (the number of states is reduced)
        player_turn = game.get_current_player()
        if player_turn == 1:
            current_board = tuple([1 - val if (val == 1 or val == 0) else val for val in current_board])

        # to favor the exploration, during the TRAINING phase,
        # we play random move both if the board is not in the Q table or if we are exploring
        if self.train is True and (self.check_map_presence(current_board) is False or random.random() < self.epsilon):
            from_pos, move = self.my_ramdom_move(current_board)  ##updated
        else:
            # end training, if the board is not in the Q table, play a random move
            if self.check_map_presence(current_board) is False:
                from_pos, move = self.my_ramdom_move(current_board)
            else:
                from_pos, move = self.choose_move(current_board)

        state = (current_board, tuple([from_pos, move]))
        self.match_moves.append(state)
        return from_pos, move

    def check_map_presence(self, board: tuple) -> bool:
        """
        check if the board (and its symmetries) are already in the Q table.
        Return True if the board is already in the Q table, False otherwise
        """
        board_90 = symmetryUtils.rotate_board_90_right(board)
        board_180 = symmetryUtils.rotate_board_90_right(board_90)
        board_270 = symmetryUtils.rotate_board_90_right(board_180)
        horizontal_mirror_board = symmetryUtils.horizontal_mirror_board(board)
        vertical_mirror_board = symmetryUtils.vertical_mirror_board(board)

        boards = [state[0] for state in self.Q]  # the player is not important anymore
        if len(boards) == 0:
            return False
        return bool(board in boards or board_90 in boards or board_180 in boards or board_270 in boards
                    or horizontal_mirror_board in boards or vertical_mirror_board in boards)

    def my_ramdom_move(self, current_board: tuple) -> tuple[tuple[int, int], Move]:
        """
        Generate a random move until a valid move is found (for the current board and player).
        """

        game_format_board = np.array(current_board).reshape(5, 5)
        while True:
            tmp_game = TrainingGame(game_format_board, self.default_player)
            available_moves_list = tmp_game.available_moves(self.default_player)
            from_pos, move = random.choice(available_moves_list)

            possible = tmp_game.move(from_pos, move, self.default_player, check_only=True)
            # heuristic condition
            good_move = False
            if possible:
                good_move = self.check_good_move(from_pos, game_format_board.tolist())

            if possible and good_move:
                return from_pos, move

    @staticmethod
    def check_good_move(from_pos: tuple, board: list) -> bool:
        """
        Check if the move is a good move, i.e. if the move is not on a cell already occupied.
        The heuristic is that the move is good if there are at least 12 empty cells of the board and the move is on an empty cell.
        """
        # empty_edges = len([True for cell in board[0] if cell == -1]) + len([True for cell in board[4] if cell == -1]) + len([True for cell in board if cell[0] == -1]) + len([True for cell in board if cell[4] == -1])
        empty_edges = len([v for row in board for v in row if v == -1])
        if empty_edges >= 12:
            if board[from_pos[1]][from_pos[0]] == 0 or board[from_pos[1]][from_pos[0]] == 1:
                return False
            else:
                return True
        else:
            return True

    def check_and_return_symmetries(self, board: tuple, set_of_board: set) -> tuple:
        if board in set_of_board:
            values = [(state, self.Q[state]) for state in self.Q if state[0] == board]
            max_current = max(values, key=lambda x: x[1])
        else:
            max_current = (None, -math.inf)
            values = []

        return max_current, values

    def choose_move(self, current_board: tuple) -> tuple[tuple[int, int], Move]:
        """
        Search for the board (and its symmetries) in the Q table to return the best move (with the highest Q value).
        """
        set_of_board = set([state[0] for state in self.Q])

        ##need to take the max of the Q values of the symmetries of the board
        max_current, values = self.check_and_return_symmetries(current_board, set_of_board)

        board_90 = symmetryUtils.rotate_board_90_right(current_board)
        max_90, values_90 = self.check_and_return_symmetries(board_90, set_of_board)

        board_180 = symmetryUtils.rotate_board_90_right(board_90)
        max_180, values_180 = self.check_and_return_symmetries(board_180, set_of_board)

        board_270 = symmetryUtils.rotate_board_90_right(board_180)
        max_270, values_270 = self.check_and_return_symmetries(board_270, set_of_board)

        horizontal_mirror_board = symmetryUtils.horizontal_mirror_board(current_board)
        max_horizontal_mirror, values_horizontal_mirror = self.check_and_return_symmetries(
            horizontal_mirror_board, set_of_board)

        vertical_mirror_board = symmetryUtils.vertical_mirror_board(current_board)
        max_vertical_mirror, values_vertical_mirror = self.check_and_return_symmetries(
            vertical_mirror_board, set_of_board)

        max_general = max([max_current, max_90, max_180, max_270, max_vertical_mirror, max_horizontal_mirror],
                          key=lambda x: x[1])

        if max_general in values:
            return max_general[0][1]
        elif max_general in values_90:
            return symmetryUtils.rotate_move_270_right(max_general[0][1])
        elif max_general in values_180:
            return symmetryUtils.rotate_move_180_right(max_general[0][1])
        elif max_general in values_270:
            return symmetryUtils.rotate_move_90_right(max_general[0][1])
        elif max_general in values_horizontal_mirror:
            return symmetryUtils.horizontal_mirror_move(max_general[0][1])
        elif max_general in values_vertical_mirror:
            return symmetryUtils.vertical_mirror_move(max_general[0][1])

    def update_Q_table(self, win: bool):
        if win:
            reward = self.reward_win
        else:
            reward = self.reward_loss
        for state in self.match_moves:
            self.update_Q_value(state, reward)

    def training(self):
        self.train = True
        generation = tqdm(range(self.training_match))
        for epoch in generation:

            self.current_match = Game()
            turn = random.randint(0, 1)
            self.match_moves = []
            players = [self, RandomPlayer()]
            begin_player = players[turn]
            winner = self.current_match.play(begin_player, players[(turn + 1) % 2])

            if winner == 0 and begin_player == self:
                self.update_Q_table(win=True)
            elif winner == 1 and begin_player != self:
                self.update_Q_table(win=True)
            else:
                self.update_Q_table(win=False)

            # and against the minmax-player
            self.current_match = Game()
            turn = random.randint(0, 1)
            self.match_moves = []
            players = [self, SymmetryMinMaxPlayer()]
            begin_player = players[turn]
            winner = self.current_match.play(begin_player, players[(turn + 1) % 2])

            if winner == 0 and begin_player == self:
                self.update_Q_table(win=True)
            elif winner == 1 and begin_player != self:
                self.update_Q_table(win=True)
            else:
                self.update_Q_table(win=False)

            if epoch % self.decrease_epsilon_epoch == 0 and epoch != 0:
                if self.epsilon > 0.1:
                    self.epsilon -= self.decrease_epsilon

        self.train = False

        # save_model(self, text=f"flipped_{self.training_match}_lr_{self.learning_rate}_df_{self.discount_factor}_e_{self.epsilon}")

    def update_Q_value(self, state, reward: int = 0):
        """
        check if the state (and its symmetries) are already in the Q table. If not, add a single entry.
        If the state is already in the Q table and the reward is not 0, update the Q value
        """
        if self.train is False:
            return
        policy = state
        policy_90 = symmetryUtils.rotate_state_90_right(policy)
        policy_180 = symmetryUtils.rotate_state_90_right(policy_90)
        policy_270 = symmetryUtils.rotate_state_90_right(policy_180)
        horizontal_mirror_policy = symmetryUtils.horizontal_mirror_state(policy)
        vertical_mirror_policy = symmetryUtils.vertical_mirror_state(policy)

        if policy not in self.Q and policy_90 not in self.Q \
                and policy_180 not in self.Q and policy_270 not in self.Q \
                and horizontal_mirror_policy not in self.Q and vertical_mirror_policy not in self.Q:

            self.Q[policy] = 0.0

        if policy_90 in self.Q:
            self.Q[policy_90] = self.Q[policy_90] + self.learning_rate * (
                        reward + self.discount_factor * self.get_max_state_Q_value(policy_90) - self.Q[policy_90])
        elif policy_180 in self.Q:
            self.Q[policy_180] = self.Q[policy_180] + self.learning_rate * (
                        reward + self.discount_factor * self.get_max_state_Q_value(policy_180) - self.Q[policy_180])
        elif policy_270 in self.Q:
            self.Q[policy_270] = self.Q[policy_270] + self.learning_rate * (
                        reward + self.discount_factor * self.get_max_state_Q_value(policy_270) - self.Q[policy_270])
        elif horizontal_mirror_policy in self.Q:
            self.Q[horizontal_mirror_policy] = self.Q[horizontal_mirror_policy] + self.learning_rate * (
                        reward + self.discount_factor * self.get_max_state_Q_value(horizontal_mirror_policy)
                        - self.Q[horizontal_mirror_policy])
        elif vertical_mirror_policy in self.Q:
            self.Q[vertical_mirror_policy] = self.Q[vertical_mirror_policy] + self.learning_rate * (
                        reward + self.discount_factor * self.get_max_state_Q_value(vertical_mirror_policy)
                        - self.Q[vertical_mirror_policy])
        else:
            self.Q[policy] = self.Q[policy] + self.learning_rate * (
                        reward + self.discount_factor * self.get_max_state_Q_value(policy) - self.Q[policy])

    def get_max_state_Q_value(self, state: tuple) -> float:
        """
        return the maximum Q value of the table state and its symmetries (if present)
        """
        table = state[0]
        table_90 = symmetryUtils.rotate_board_90_right(table)
        table_180 = symmetryUtils.rotate_board_90_right(table_90)
        table_270 = symmetryUtils.rotate_board_90_right(table_180)
        horizontal_mirror_table = symmetryUtils.horizontal_mirror_board(table)
        vertical_mirror_table = symmetryUtils.vertical_mirror_board(table)

        table_values = [self.Q[(tab, move)] for tab, move, in self.Q if (tab == table or tab == table_90 or tab == table_180 or tab == table_270 or tab == horizontal_mirror_table or tab == vertical_mirror_table)]
        if len(table_values) == 0:
            return 0
        return max(table_values)


def create_symmetry_agent(tmp: Player) -> Player:
    training_match = tmp.training_match
    reward_win = tmp.reward_win
    reward_loss = tmp.reward_loss

    tmp_agent = MultiSymmetryAgent(training_match=training_match, reward_win=reward_win, reward_loss=reward_loss)
    tmp_agent.training()
    return tmp_agent


class SpeedUpSymmetryAgent:
    """
    class that creates N agents and train them in parallel using the Q-learning algorithm, at the end of the training
    all the Q tables are merged into a single one using a weighted sum of the Q values
    """

    def __init__(self, agents_number: int, trainig_period: int = 1000, reward_win: int = 1,
                 reward_loss: int = -3) -> MultiSymmetryAgent:
        self.agents_number = agents_number
        self.trainig_period = trainig_period
        self.reward_win = reward_win
        self.reward_loss = reward_loss

    def training(self):
        agents = [MultiSymmetryAgent(self.trainig_period, self.reward_win, self.reward_loss) for _ in
                  range(self.agents_number)]
        with mp.Pool() as pool:
            agents = pool.map(create_symmetry_agent, agents)
        ### merge the Q tables of the agents, if the same state is present in more than one Q table, the Q value is the average of the Q values
        ### it is necessary to check also the symmetries of the state
        Q = defaultdict(float)
        frequency_presence = defaultdict(int)
        division = self.agents_number // 2
        sum_Q = 0
        for agent in agents:
            # print(len(agent.Q))
            sum_Q += len(agent.Q)
            for state in agent.Q:
                state_90 = symmetryUtils.rotate_state_90_right(state)
                state_180 = symmetryUtils.rotate_state_90_right(state_90)
                state_270 = symmetryUtils.rotate_state_90_right(state_180)
                state_horizontal_mirror = symmetryUtils.horizontal_mirror_state(state)
                state_vertical_mirror = symmetryUtils.vertical_mirror_state(state)

                if state in Q:
                    Q[state] += agent.Q[state] / division
                    frequency_presence[state] += 1
                elif state_90 in Q:
                    Q[state_90] += agent.Q[state] / division
                    frequency_presence[state_90] += 1
                elif state_180 in Q:
                    Q[state_180] += agent.Q[state] / division
                    frequency_presence[state_180] += 1
                elif state_270 in Q:
                    Q[state_270] += agent.Q[state] / division
                    frequency_presence[state_270] += 1
                elif state_horizontal_mirror in Q:
                    Q[state_horizontal_mirror] += agent.Q[state] / division
                    frequency_presence[state_horizontal_mirror] += 1
                elif state_vertical_mirror in Q:
                    Q[state_vertical_mirror] += agent.Q[state] / division
                    frequency_presence[state_vertical_mirror] += 1
                else:
                    Q[state] = agent.Q[state] / division
                    frequency_presence[state] = 1
        ##voting apporach
        # for state in Q:
        #     Q[state] = Q[state] / frequency_presence[state]
        print("Sum Q table: ", sum_Q)
        print("Merged Q table: ", len(Q))
        mixed_agent = MultiSymmetryAgent(self.trainig_period, self.reward_win, self.reward_loss)
        mixed_agent.Q = Q
        mixed_agent.train = False
        save_model(mixed_agent, text=f"mixed_addedmirror_{self.agents_number}_agents_{self.trainig_period}_for_agent")
        return mixed_agent
