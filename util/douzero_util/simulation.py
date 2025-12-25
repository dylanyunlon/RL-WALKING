"""
Modified simulation.py for LLM4CardGame Training Data Generation
================================================================

This version records COMPLETE game trajectories including:
- hand_cards: Current player's hand cards
- legal_actions: All legal moves available
- action: The action taken by the player
- position: Player's role (landlord, landlord_up, landlord_down)
- other game state information

Usage:
    Replace the original util/douzero_util/simulation.py with this file,
    or use the generate_training_data() function directly.
"""

import multiprocessing as mp
import pickle
import time
import os
import json
import random
import numpy as np
from copy import deepcopy

from douzero.env.game import GameEnv
from util.llm_client import llm_function
from util.prompt_util import prompt_function_dou_dizhu, out_parse_function_dou_dizhu
from util.llm_config import model_config
from .llm_agent import LLMAgent
from .rlcard_agent import RLCardAgent
from .random_agent import RandomAgent
from .deep_agent import DeepAgent


def load_card_play_models(card_play_model_path_dict):
    """Load player models based on configuration."""
    players = {}

    for position in ['landlord', 'landlord_up', 'landlord_down']:
        model_path = card_play_model_path_dict[position]
        
        if model_path in model_config:
            players[position] = LLMAgent(
                llm_function, 
                model_config[model_path], 
                prompt_function_dou_dizhu, 
                out_parse_function_dou_dizhu
            )
        elif model_path == 'rlcard':  
            players[position] = RLCardAgent(position)
        elif model_path == 'random':
            players[position] = RandomAgent()
        elif model_path == 'pre':
            pre_path = f'../DouZero/baselines/douzero_WP/{position}.ckpt'
            players[position] = DeepAgent(position, pre_path)
        else:
            players[position] = DeepAgent(position, model_path)
    
    return players


class TrajectoryGameEnv(GameEnv):
    """
    Extended GameEnv that records complete game trajectories for training.
    """
    
    def __init__(self, players, log_path=None):
        super().__init__(players)
        self.log_path = log_path
        self.trajectory = []  # Store trajectory for current game
        self.all_trajectories = []  # Store all game trajectories
        
    def log_data(self, data):
        """Original log_data method for compatibility."""
        if self.log_path:
            with open(self.log_path, 'a') as f:
                f.write(str(data) + '\n')
    
    def step(self):
        """
        Extended step method that records the game state before taking action.
        """
        # Get current game state BEFORE the action
        current_position = self.acting_player_position
        infoset = self.game_infoset
        
        # Get hand cards and legal actions
        hand_cards = list(infoset.player_hand_cards) if hasattr(infoset, 'player_hand_cards') else []
        legal_actions = list(infoset.legal_actions) if hasattr(infoset, 'legal_actions') else []
        
        # Get other players' card counts
        num_cards_left = {
            'landlord': len(self.info_sets['landlord'].player_hand_cards),
            'landlord_up': len(self.info_sets['landlord_up'].player_hand_cards),
            'landlord_down': len(self.info_sets['landlord_down'].player_hand_cards),
        }
        
        # Get last move
        last_move = self.get_last_move() if hasattr(self, 'get_last_move') else []
        
        # Get action from player
        action = self.players[self.acting_player_position].act(self.game_infoset)
        
        # Record the step data
        step_data = {
            'position': current_position,
            'hand_cards': [int(c) for c in hand_cards],
            'legal_actions': [[int(c) for c in a] for a in legal_actions],
            'action': [int(c) for c in action] if action else [],
            'last_move': [int(c) for c in last_move] if last_move else [],
            'num_cards_left': num_cards_left,
            'played_cards': {k: [int(c) for c in v] for k, v in self.played_cards.items()},
            'bomb_num': self.bomb_num,
            'three_landlord_cards': [int(c) for c in self.three_landlord_cards] if self.three_landlord_cards else [],
        }
        
        # Only record if hand_cards and legal_actions are not empty
        if hand_cards and legal_actions:
            self.trajectory.append(step_data)
        
        # Continue with original step logic
        if len(action) > 0:
            self.last_pid = self.acting_player_position

        from douzero.env.game import bombs
        if action in bombs:
            self.bomb_num += 1

        self.last_move_dict[self.acting_player_position] = action.copy()
        self.card_play_action_seq.append(action)
        self.update_acting_player_hand_cards(action)
        self.played_cards[self.acting_player_position] += action

        if self.acting_player_position == 'landlord' and \
                len(action) > 0 and \
                len(self.three_landlord_cards) > 0:
            for card in action:
                if len(self.three_landlord_cards) > 0:
                    if card in self.three_landlord_cards:
                        self.three_landlord_cards.remove(card)
                else:
                    break

        self.game_done()
        if not self.game_over:
            self.get_acting_player_position()
            self.game_infoset = self.get_infoset()
    
    def get_trajectory(self):
        """Get the trajectory for the current game."""
        return self.trajectory
    
    def reset(self):
        """Reset for next game, save current trajectory."""
        if self.trajectory:
            self.all_trajectories.append({
                'trajectory': self.trajectory,
                'winner': getattr(self, 'winner', None),
            })
        self.trajectory = []
        
        # Reset parent class state
        self.card_play_action_seq = []
        self.three_landlord_cards = None
        self.game_over = False
        self.acting_player_position = None
        self.player_utility_dict = None
        self.last_move_dict = {'landlord': [], 'landlord_up': [], 'landlord_down': []}
        self.played_cards = {'landlord': [], 'landlord_up': [], 'landlord_down': []}
        self.last_move = []
        self.last_two_moves = []
        self.bomb_num = 0
        self.last_pid = 'landlord'


def mp_simulate_with_trajectory(card_play_data_list, card_play_model_path_dict, q, log_dir, seed, worker_id):
    """
    Modified simulation that records complete trajectories.
    """
    seed = seed + worker_id * 100
    np.random.seed(seed)
    random.seed(seed)

    players = load_card_play_models(card_play_model_path_dict)

    time_str = time.strftime('-%Y%m%d%H%M%S', time.localtime(time.time()))
    log_path = os.path.join(log_dir, f'log{time_str}-{worker_id}.txt')
    trajectory_path = os.path.join(log_dir, f'trajectory{time_str}-{worker_id}.jsonl')
    
    env = TrajectoryGameEnv(players, log_path)
    
    for idx, card_play_data in enumerate(card_play_data_list):
        env.log_data(card_play_model_path_dict)
        env.card_play_init(card_play_data)
        
        while not env.game_over:
            env.step()
        
        env.log_data({'winner': env.winner})
        env.reset()

    # Save trajectories to JSONL file
    with open(trajectory_path, 'w') as f:
        for game_data in env.all_trajectories:
            f.write(json.dumps(game_data, ensure_ascii=False) + '\n')

    # Request data for statistics
    count_records = [[0, 0] for _ in range(3)]     
    for pid, position in enumerate(['landlord', 'landlord_up', 'landlord_down']):
        if hasattr(players[position], 'request_count'):
            count_records[pid][0] = players[position].request_count
            count_records[pid][1] = players[position].correct_count
    
    q.put((
        env.num_wins['landlord'],
        env.num_wins['farmer'],
        env.num_scores['landlord'],
        env.num_scores['farmer'],
        count_records,
        len(env.all_trajectories)  # Number of games with trajectories
    ))


def mp_simulate(card_play_data_list, card_play_model_path_dict, q, log_dir, seed, worker_id):
    """Original mp_simulate for backward compatibility."""
    seed = seed + worker_id * 100
    np.random.seed(seed)
    random.seed(seed)

    players = load_card_play_models(card_play_model_path_dict)

    time_str = time.strftime('-%Y%m%d%H%M%S', time.localtime(time.time()))
    log_path = os.path.join(log_dir, f'log{time_str}-{worker_id}.txt')
    env = GameEnv(players)
    env.log_path = log_path
    
    # Add log_data method to env
    def log_data(data):
        with open(log_path, 'a') as f:
            f.write(str(data) + '\n')
    env.log_data = log_data
    
    for idx, card_play_data in enumerate(card_play_data_list):
        env.log_data(card_play_model_path_dict)
        env.card_play_init(card_play_data)
        while not env.game_over:
            env.step()
        env.log_data({'winner': env.winner})
        env.reset()

    count_records = [[0, 0] for _ in range(3)]     
    for pid, position in enumerate(['landlord', 'landlord_up', 'landlord_down']):
        if hasattr(players[position], 'request_count'):
            count_records[pid][0] = players[position].request_count
            count_records[pid][1] = players[position].correct_count
    q.put((env.num_wins['landlord'],
           env.num_wins['farmer'],
           env.num_scores['landlord'],
           env.num_scores['farmer'],
           count_records
         ))


def data_allocation_per_worker(card_play_data_list, num_workers):
    """Distribute data across workers."""
    card_play_data_list_each_worker = [[] for k in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        card_play_data_list_each_worker[idx % num_workers].append(data)
    return card_play_data_list_each_worker


def evaluate(landlord, landlord_up, landlord_down, eval_data, num_workers, seed, log_dir):
    """Original evaluate function for backward compatibility."""
    if landlord in model_config:
        num_workers = min(num_workers, model_config[landlord]['call_func']['num_workers'])
    if landlord_up in model_config:
        num_workers = min(num_workers, model_config[landlord_up]['call_func']['num_workers'])
    if landlord_down in model_config:
        num_workers = min(num_workers, model_config[landlord_down]['call_func']['num_workers'])
    
    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    card_play_data_list_each_worker = data_allocation_per_worker(
        card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        'landlord': landlord,
        'landlord_up': landlord_up,
        'landlord_down': landlord_down}

    num_landlord_wins = 0
    num_farmer_wins = 0
    num_landlord_scores = 0
    num_farmer_scores = 0
    all_count_records = [[0, 0] for _ in range(3)]  

    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []
    for worker_id, card_play_data in enumerate(card_play_data_list_each_worker):
        p = ctx.Process(
                target=mp_simulate,
                args=(card_play_data, card_play_model_path_dict, q, log_dir, seed, worker_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for i in range(num_workers):
        result = q.get()
        num_landlord_wins += result[0]
        num_farmer_wins += result[1]
        num_landlord_scores += result[2]
        num_farmer_scores += result[3]
        result_count = result[4]
        for j in range(len(result_count)):
            all_count_records[j][0] += result_count[j][0]
            all_count_records[j][1] += result_count[j][1]

    num_total_wins = num_landlord_wins + num_farmer_wins
    print('Model list:')
    print(card_play_model_path_dict)
    print('WP results:')
    print('landlord : Farmers - {:.3f} : {:.3f}'.format(
        num_landlord_wins / num_total_wins, num_farmer_wins / num_total_wins))
    print('ADP results:')
    print('landlord : Farmers - {:.3f} : {:.3f}'.format(
        num_landlord_scores / num_total_wins, 2 * num_farmer_scores / num_total_wins)) 
    models = [card_play_model_path_dict[position] for position in ['landlord', 'landlord_up', 'landlord_down']]
    print('Format Accuracy results:')
    for position, reward in enumerate(all_count_records):
        if reward[0] == 0:
            reward = -1
        else:
            reward = reward[1] / reward[0]
        print(position, models[position], f'{reward:.3f}')


def generate_training_data(landlord, landlord_up, landlord_down, eval_data, num_workers, seed, log_dir):
    """
    Generate training data with complete trajectories.
    
    This is the main function for generating SFT training data.
    It records hand_cards, legal_actions, and actions for each step.
    """
    print(f"Generating training data with trajectories...")
    print(f"  Landlord: {landlord}")
    print(f"  Farmers: {landlord_up}, {landlord_down}")
    print(f"  Output: {log_dir}")
    
    if landlord in model_config:
        num_workers = min(num_workers, model_config[landlord]['call_func']['num_workers'])
    if landlord_up in model_config:
        num_workers = min(num_workers, model_config[landlord_up]['call_func']['num_workers'])
    if landlord_down in model_config:
        num_workers = min(num_workers, model_config[landlord_down]['call_func']['num_workers'])
    
    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)
    
    print(f"  Loaded {len(card_play_data_list)} games from {eval_data}")

    card_play_data_list_each_worker = data_allocation_per_worker(
        card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        'landlord': landlord,
        'landlord_up': landlord_up,
        'landlord_down': landlord_down}

    os.makedirs(log_dir, exist_ok=True)

    num_landlord_wins = 0
    num_farmer_wins = 0
    num_landlord_scores = 0
    num_farmer_scores = 0
    total_trajectories = 0
    all_count_records = [[0, 0] for _ in range(3)]  

    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []
    
    for worker_id, card_play_data in enumerate(card_play_data_list_each_worker):
        p = ctx.Process(
            target=mp_simulate_with_trajectory,
            args=(card_play_data, card_play_model_path_dict, q, log_dir, seed, worker_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for i in range(num_workers):
        result = q.get()
        num_landlord_wins += result[0]
        num_farmer_wins += result[1]
        num_landlord_scores += result[2]
        num_farmer_scores += result[3]
        result_count = result[4]
        total_trajectories += result[5]
        for j in range(len(result_count)):
            all_count_records[j][0] += result_count[j][0]
            all_count_records[j][1] += result_count[j][1]

    num_total_wins = num_landlord_wins + num_farmer_wins
    
    print('\n=== Training Data Generation Results ===')
    print('Model list:')
    print(card_play_model_path_dict)
    print(f'\nTotal games with trajectories: {total_trajectories}')
    print('WP results:')
    print('landlord : Farmers - {:.3f} : {:.3f}'.format(
        num_landlord_wins / num_total_wins, num_farmer_wins / num_total_wins))
    print('ADP results:')
    print('landlord : Farmers - {:.3f} : {:.3f}'.format(
        num_landlord_scores / num_total_wins, 2 * num_farmer_scores / num_total_wins))
    
    # List generated trajectory files
    trajectory_files = [f for f in os.listdir(log_dir) if f.startswith('trajectory') and f.endswith('.jsonl')]
    print(f'\nGenerated {len(trajectory_files)} trajectory files in {log_dir}')
    
    return trajectory_files


if __name__ == '__main__':
    # Test the trajectory generation
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--landlord', type=str, default='pre')
    parser.add_argument('--landlord_up', type=str, default='rlcard')
    parser.add_argument('--landlord_down', type=str, default='rlcard')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--seed', type=int, default=43)
    
    args = parser.parse_args()
    
    generate_training_data(
        args.landlord,
        args.landlord_up,
        args.landlord_down,
        args.eval_data,
        args.num_workers,
        args.seed,
        args.log_dir
    )