from copy import deepcopy
from game import Move


def rotate_state_90_right(state: tuple) -> tuple:
    ## the board is a tuple of 25 elements
    return rotate_board_90_right(state[0]), rotate_move_90_right(state[1])


def rotate_board_90_right(table: tuple) -> tuple:
    '''
    Rotate the Quixo game state by 90 degrees.

    Parameters:
    - state (tuple): The current game state represented as a tuple of 25 elements.

    Returns:
    - tuple: The rotated game state.
        '''
    board = deepcopy(table)
    new_board = list([-1] * 25)
    for i in range(25):
        row = i // 5
        col = i % 5
        new_index = 5 * col + (4 - row)
        new_board[new_index] = board[i]
    return tuple(new_board)


def rotate_move_90_right(mov: tuple) -> tuple[tuple[int, int], Move]:
    current_mov = mov[1]
    new_move = None

    if current_mov == Move.TOP:
        new_move = Move.RIGHT
    elif current_mov == Move.BOTTOM:
        new_move = Move.LEFT
    elif current_mov == Move.LEFT:
        new_move = Move.TOP
    elif current_mov == Move.RIGHT:
        new_move = Move.BOTTOM

    position = mov[0]
    # row = position[0]
    # col = position[1]
    row = position[1]
    col = position[0]
    new_index = 5 * col + (4 - row)
    new_position = (new_index % 5, new_index // 5)

    return new_position, new_move


def rotate_state_180_right(state: tuple) -> tuple:
    ## the board is a tuple of 25 elements
    return rotate_board_180_right(state[0]), rotate_move_180_right(state[1])


def rotate_board_180_right(table: tuple) -> tuple:
    return tuple(table[i] for i in range(24, -1, -1))


def rotate_move_180_right(mov: tuple) -> tuple[tuple[int, int], Move]:
    current_mov = mov[1]
    new_move = None

    if current_mov == Move.TOP:
        new_move = Move.BOTTOM
    elif current_mov == Move.BOTTOM:
        new_move = Move.TOP
    elif current_mov == Move.LEFT:
        new_move = Move.RIGHT
    elif current_mov == Move.RIGHT:
        new_move = Move.LEFT

    position = mov[0]
    X = 5 - position[0] - 1
    Y = 5 - position[1] - 1
    new_position = (X, Y)

    return new_position, new_move


def rotate_state_270_right(state: tuple) -> tuple:
    ## the board is a tuple of 25 elements
    return rotate_board_270_right(state[0]), rotate_move_270_right(state[1])


def rotate_board_270_right(table: tuple) -> tuple:
    rotation = rotate_board_180_right(table)
    return rotate_board_90_right(rotation)


def rotate_move_270_right(mov: tuple) -> tuple[tuple[int, int], Move]:
    rotation = rotate_move_180_right(mov)
    return rotate_move_90_right(rotation)


def vertical_mirror_state(state: tuple) -> tuple:
    ## the board is a tuple of 25 elements
    return vertical_mirror_board(state[0]), vertical_mirror_move(state[1])


def vertical_mirror_board(table: tuple) -> tuple:
    board = deepcopy(table)
    mirror = tuple(board[i:i + 5][::1] for i in range(20, -1, -5))
    tmp = ()
    for i in mirror:
        tmp += i
    return tmp


def vertical_mirror_move(mov: tuple) -> tuple[tuple[int, int], Move]:
    current_mov = mov[1]
    new_move = None

    if current_mov == Move.TOP:
        new_move = Move.BOTTOM
    elif current_mov == Move.BOTTOM:
        new_move = Move.TOP
    elif current_mov == Move.LEFT:
        new_move = Move.LEFT
    elif current_mov == Move.RIGHT:
        new_move = Move.RIGHT

    position = mov[0]
    X = position[0]
    Y = 5 - position[1] - 1
    new_position = (X, Y)

    return new_position, new_move


def horizontal_mirror_state(state: tuple) -> tuple:
    ## the board is a tuple of 25 elements
    return horizontal_mirror_board(state[0]), horizontal_mirror_move(state[1])


def horizontal_mirror_board(table: tuple) -> tuple:
    board = deepcopy(table)
    mirrored_board_state = tuple(board[i:i + 5][::-1] for i in range(0, 25, 5))
    tmp = ()
    for i in mirrored_board_state:
        tmp += i
    return tmp


def horizontal_mirror_move(mov: tuple) -> tuple[tuple[int, int], Move]:
    '''
    Produce the horizontal mirror of the move
    '''
    current_mov = mov[1]
    new_move = None

    if current_mov == Move.TOP:
        new_move = Move.TOP
    elif current_mov == Move.BOTTOM:
        new_move = Move.BOTTOM
    elif current_mov == Move.LEFT:
        new_move = Move.RIGHT
    elif current_mov == Move.RIGHT:
        new_move = Move.LEFT

    position = mov[0]
    X = 5 - position[0] - 1
    Y = position[1]
    new_position = (X, Y)

    return new_position, new_move
