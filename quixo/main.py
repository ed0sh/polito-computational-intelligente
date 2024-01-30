import random

from tqdm import tqdm
import numpy as np
from utils import RandomPlayer, load_model

from RLAgent import SpeedUpSymmetryAgent,MultiSymmetryAgent
from game import Game, Move, Player
import utils
import multiprocessing as mp
from MinMaxPlayer import SymmetryMinMaxPlayer


def play_game(tmp : tuple) -> int : 
    '''
    Return the number of win of the second player inserted in the tuple
    '''
    count = 0
    player0 = tmp[0]
    player1 = tmp[1]
    for i in tqdm(range(10)):
        g = Game()    
        if random.random() < 0.5:
            count += 1 if g.play(player0, player1) == 1 else 0
        else:
            count += 1 if g.play(player1, player0) == 0 else 0
    return count



if __name__ == '__main__':
    
    #player_with_RL = SpeedUpSymmetryAgent(agents_number=8, trainig_period=25).training()
    player_with_RL = utils.load_model('models/agent-mixed_addedmirror_8_agents_10000_for_agent.pkl')
    min_max_player = SymmetryMinMaxPlayer()


    count = 0
    games_x_10 = 10
    player_to_process = [(RandomPlayer(),player_with_RL)]*games_x_10
    with mp.Pool() as pool:
        results = pool.map(play_game,player_to_process)
    
    print(sum(results))
