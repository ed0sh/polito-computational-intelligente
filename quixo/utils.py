from game import Game, Player, Move
import pickle
import random


def save_model(model: Player, text: str = None):
    # Serialize the object and write it to a file
    with open(f'models/agent-{text}.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str) -> Player:
    # Load the model from a file
    with open(path, 'rb') as f:
        loaded_instance = pickle.load(f)
    return loaded_instance


def from_board_to_state_element(board) -> tuple:
    state_element = list()
    [state_element.extend(list(x)) for x in board]
    return tuple(state_element)


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move
