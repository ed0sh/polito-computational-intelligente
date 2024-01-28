from copy import deepcopy

import numpy as np

from game import Game
from game import Move


class TrainingGame(Game):
    def __init__(self, board: np.array, current_player_idx: int) -> None:
        super().__init__()
        self._board = board
        self.current_player_idx = current_player_idx

    def move(self, from_pos: tuple[int, int], slide: Move, player_id: int, check_only: bool = False) -> bool:
        return self.__move(from_pos, slide, player_id, check_only)

    def available_moves(self, player_idx: int) -> list[tuple[tuple[int, int], Move]]:
        available_moves = []

        playable_positions = np.bitwise_or(self._board == player_idx, self._board == -1)
        for (y, x), value in np.ndenumerate(playable_positions):
            # If it's on a border and I can play that position
            if (x == 0 or x == 4 or y == 0 or y == 4) and value:
                for move in [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]:
                    if self.move((x, y), move, player_idx, check_only=True):
                        available_moves.append(((x, y), move))

        return available_moves

    def get_intermediate_final_state_value(self, player_idx: int) -> int:
        '''
        Returns the intermediate final state value for the given player by calculating the maximum number of elements
        in each row and column or diagonal, subtracted by the opponent elements in that line
        :param player_idx: Player index
        :return: The total
        '''
        max_elements = -1
        max_el_coord = (None, None)

        player_moves = (self._board == player_idx).astype(int)
        # for each row
        for x in range(player_moves.shape[0]):
            if sum(player_moves[x, :]) > max_elements:
                max_elements = sum(player_moves[x, :])
                max_el_coord = (x, None)
        # for each column
        for y in range(player_moves.shape[1]):
            if sum(player_moves[:, y]) > max_elements:
                max_elements = sum(player_moves[:, y])
                max_el_coord = (None, y)
        # principal diagonal
        if sum([player_moves[x, x] for x in range(player_moves.shape[0])]) > max_elements:
            max_elements = sum([player_moves[x, x] for x in range(player_moves.shape[0])])
            max_el_coord = (0, 0)
        # secondary diagonal
        if sum([player_moves[x, -(x + 1)] for x in range(player_moves.shape[0])]) > max_elements:
            max_elements = sum([player_moves[x, -(x + 1)] for x in range(player_moves.shape[0])])
            max_el_coord = (0, 4)

        opponent_moves = (self._board == abs(1 - player_idx)).astype(int)
        opponent_max = 0
        if max_el_coord[0] is not None and max_el_coord[1] is None:
            opponent_max = sum(opponent_moves[max_el_coord[0], :])
        elif max_el_coord[0] is None and max_el_coord[1] is not None:
            opponent_max = sum(opponent_moves[:, max_el_coord[1]])
        elif max_el_coord[0] == 0 and max_el_coord[1] == 0:
            opponent_max = sum([opponent_moves[x, x] for x in range(opponent_moves.shape[0])])
        elif max_el_coord[0] == 0 and max_el_coord[1] == 4:
            opponent_max = sum([opponent_moves[x, -(x + 1)] for x in range(opponent_moves.shape[0])])

        return int(max_elements - opponent_max)

    def __move(self, from_pos: tuple[int, int], slide: Move, player_id: int, check_only: bool = False) -> bool:
        '''Perform a move'''
        if player_id > 2:
            return False
        # Oh God, Numpy arrays
        prev_value = deepcopy(self._board[(from_pos[1], from_pos[0])])
        acceptable = self.__take((from_pos[1], from_pos[0]), player_id, check_only)
        if acceptable:
            acceptable = self.__slide((from_pos[1], from_pos[0]), slide, check_only)
            if not acceptable:
                self._board[(from_pos[1], from_pos[0])] = deepcopy(prev_value)
        return acceptable

    def __take(self, from_pos: tuple[int, int], player_id: int, check_only: bool = False) -> bool:
        '''Take piece'''
        # acceptable only if in border
        acceptable: bool = (
            # check if it is in the first row
            (from_pos[0] == 0 and from_pos[1] < 5)
            # check if it is in the last row
            or (from_pos[0] == 4 and from_pos[1] < 5)
            # check if it is in the first column
            or (from_pos[1] == 0 and from_pos[0] < 5)
            # check if it is in the last column
            or (from_pos[1] == 4 and from_pos[0] < 5)
            # and check if the piece can be moved by the current player
        ) and (self._board[from_pos] < 0 or self._board[from_pos] == player_id)
        if acceptable and not check_only:
            self._board[from_pos] = player_id
        return True

    def __slide(self, from_pos: tuple[int, int], slide: Move, check_only: bool = False) -> bool:
        '''Slide the other pieces'''
        # define the corners
        SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
        # if the piece position is not in a corner
        if from_pos not in SIDES:
            # if it is at the TOP, it can be moved down, left or right
            acceptable_top: bool = from_pos[0] == 0 and (
                slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is at the BOTTOM, it can be moved up, left or right
            acceptable_bottom: bool = from_pos[0] == 4 and (
                slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is on the LEFT, it can be moved up, down or right
            acceptable_left: bool = from_pos[1] == 0 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
            )
            # if it is on the RIGHT, it can be moved up, down or left
            acceptable_right: bool = from_pos[1] == 4 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
            )
        # if the piece position is in a corner
        else:
            # if it is in the upper left corner, it can be moved to the right and down
            acceptable_top: bool = from_pos == (0, 0) and (
                slide == Move.BOTTOM or slide == Move.RIGHT)
            # if it is in the lower left corner, it can be moved to the right and up
            acceptable_left: bool = from_pos == (4, 0) and (
                slide == Move.TOP or slide == Move.RIGHT)
            # if it is in the upper right corner, it can be moved to the left and down
            acceptable_right: bool = from_pos == (0, 4) and (
                slide == Move.BOTTOM or slide == Move.LEFT)
            # if it is in the lower right corner, it can be moved to the left and up
            acceptable_bottom: bool = from_pos == (4, 4) and (
                slide == Move.TOP or slide == Move.LEFT)
        # check if the move is acceptable
        acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
        # if it is
        if acceptable and not check_only:
            # take the piece
            piece = self._board[from_pos]
            # if the player wants to slide it to the left
            if slide == Move.LEFT:
                # for each column starting from the column of the piece and moving to the left
                for i in range(from_pos[1], 0, -1):
                    # copy the value contained in the same row and the previous column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i - 1)]
                # move the piece to the left
                self._board[(from_pos[0], 0)] = piece
            # if the player wants to slide it to the right
            elif slide == Move.RIGHT:
                # for each column starting from the column of the piece and moving to the right
                for i in range(from_pos[1], self._board.shape[1] - 1, 1):
                    # copy the value contained in the same row and the following column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i + 1)]
                # move the piece to the right
                self._board[(from_pos[0], self._board.shape[1] - 1)] = piece
            # if the player wants to slide it upward
            elif slide == Move.TOP:
                # for each row starting from the row of the piece and going upward
                for i in range(from_pos[0], 0, -1):
                    # copy the value contained in the same column and the previous row
                    self._board[(i, from_pos[1])] = self._board[(
                        i - 1, from_pos[1])]
                # move the piece up
                self._board[(0, from_pos[1])] = piece
            # if the player wants to slide it downward
            elif slide == Move.BOTTOM:
                # for each row starting from the row of the piece and going downward
                for i in range(from_pos[0], self._board.shape[0] - 1, 1):
                    # copy the value contained in the same column and the following row
                    self._board[(i, from_pos[1])] = self._board[(
                        i + 1, from_pos[1])]
                # move the piece down
                self._board[(self._board.shape[0] - 1, from_pos[1])] = piece
        return acceptable

    def pretty_print(self):
        '''Prints the board in a pretty format'''
        for r in range(5):
            print('|', end='')
            for c in range(5):
                if self._board[r][c] == 0:
                    print('✖️|', end='')
                elif self._board[r][c] == 1:
                    print('⭕|', end='')
                else:
                    print(f'  |', end="")
            if r < 4:
                print('\n-----------------')
            else:
                print()
        print()
