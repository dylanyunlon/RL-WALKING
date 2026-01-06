''' An example of evluating the trained models in RLCard
'''
import os
import argparse
import json
import time
from datetime import datetime
import multiprocessing as mp

import rlcard
from rlcard.agents import (
    DQNAgent,
    RandomAgent,
)
from rlcard.utils import (
    get_device,
    set_seed,
    # tournament,
    # reorganize
)

from util.rlcard_util.llm_agent import LLMAgent
from util.llm_client import llm_function
from util.prompt_util import prompt_function_leduc_holdem, prompt_function_limit_holdem
from util.prompt_util import out_parse_function_leduc_holdem, out_parse_function_limit_holdem
from util.prompt_util import prompt_function_uno, out_parse_function_uno
from util.prompt_util import prompt_function_nolimit_holdem, out_parse_function_nolimit_holdem
from util.prompt_util import prompt_function_gin_rummy, out_parse_function_gin_rummy
from util.prompt_util import str_to_leduc_holdem_action, str_to_limit_holdem_action, str_to_uno_action, str_to_nolimit_holdem_action, str_to_gin_rummy_action
from util.llm_config import model_config

def get_template(name):
    if name == 'leduc-holdem':
        return prompt_function_leduc_holdem, out_parse_function_leduc_holdem, str_to_leduc_holdem_action
    elif name == 'limit-holdem':
        return prompt_function_limit_holdem, out_parse_function_limit_holdem, str_to_limit_holdem_action
    elif name == 'uno':
        return prompt_function_uno, out_parse_function_uno, str_to_uno_action
    elif name == 'no-limit-holdem':
        return prompt_function_nolimit_holdem, out_parse_function_nolimit_holdem, str_to_nolimit_holdem_action
    elif name == 'gin-rummy':
        return prompt_function_gin_rummy, out_parse_function_gin_rummy, str_to_gin_rummy_action

from rlcard.games.mahjong.card import MahjongCard
from rlcard.games.nolimitholdem.round import Action
from rlcard.games.nolimitholdem.game import Stage
import numpy as np

class RLCardEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, MahjongCard):
            return obj.get_str()
        if isinstance(obj, Action):
            return obj.name
        if isinstance(obj, Stage):
            return obj.name
        # Handle numpy types
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        # Handle Gin Rummy ActionEvent
        if hasattr(obj, '__str__') and 'ActionEvent' in type(obj).__name__:
            return str(obj)
        return super().default(obj)
        
pre_dict = {
    'leduc-holdem': '/workspace/ww/rl-card-results-1226/leduc-holdem/checkpoint_dqn.pt',
    'limit-holdem': '/workspace/ww/rl-card-results-1226/limit-holdem/checkpoint_dqn.pt',
    'no-limit-holdem': '/workspace/ww/rl-card-results-1226/no-limit-holdem/checkpoint_dqn.pt',
}

def load_model(model_path, env=None, position=None, device=None):
    if model_path == 'dqn':
        import torch
        # agent = torch.load(model_path, map_location=device)
        pre_path = pre_dict[env.name]
        agent = DQNAgent.from_checkpoint(checkpoint=torch.load(pre_path, map_location=device))
        agent.set_device(device)
    elif model_path in model_config:
        prompt_func, out_func, convert_func = get_template(env.name)
        agent = LLMAgent(llm_function, model_config[model_path], prompt_func, out_func, convert_func)
    elif os.path.isfile(model_path):  # Torch model
        import torch
        # agent = torch.load(model_path, map_location=device)
        agent = DQNAgent.from_checkpoint(checkpoint=torch.load(model_path, map_location=device))
        agent.set_device(device)
    elif os.path.isdir(model_path):  # CFR model
        from rlcard.agents import CFRAgent
        agent = CFRAgent(env, model_path)
        agent.load()
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model_path).agents[position]
    
    return agent

def tournament(env, num):
    ''' Evaluate he performance of the agents in the environment

    Args:
        env (Env class): The environment to be evaluated.
        num (int): The number of games to play.

    Returns:
        A list of avrage payoffs for each player
    '''
    all_trajectories = []
    all_rewards = []
    payoffs = [0 for _ in range(env.num_players)]
    counter = 0
    while counter < num:
        _trajectories, _payoffs = env.run(is_training=False)
        all_trajectories.append(_trajectories)
        all_rewards.append(_payoffs)
        if isinstance(_payoffs, list):
            for _p in _payoffs:
                for i, _ in enumerate(payoffs):
                    payoffs[i] += _p[i]
                counter += 1
        else:
            for i, _ in enumerate(payoffs):
                payoffs[i] += _payoffs[i]
            counter += 1
    # for i, _ in enumerate(payoffs):
    #     payoffs[i] /= counter
    return payoffs, all_trajectories, all_rewards


def _convert_action_to_str(action, env):
    """
    Convert action to string representation based on environment type.
    
    Args:
        action: The action to convert (can be int, str, numpy.int64, or other types)
        env: The RLCard environment
    
    Returns:
        str: String representation of the action
    """
    # MahjongCard type
    if isinstance(action, MahjongCard):
        return action.get_str()
    
    # For other types (like Action enum from nolimitholdem)
    if isinstance(action, Action):
        return action.name
    
    # Convert string numbers to int for gin-rummy
    # e.g., "2" -> 2 -> "draw_card"
    action_int = None
    if isinstance(action, str):
        # Check if it's a numeric string that needs conversion
        if action.isdigit():
            action_int = int(action)
        else:
            # Already a proper action string (e.g., "r-9" for UNO, "call" for poker)
            return action
    elif isinstance(action, (int, np.integer)):  # Handle both Python int and numpy integers
        action_int = int(action)
    
    # Convert integer action ID based on game type
    if action_int is not None and env is not None:
        if env.name == 'uno':
            from rlcard.games.uno.utils import ACTION_LIST
            if 0 <= action_int < len(ACTION_LIST):
                return ACTION_LIST[action_int]
        elif env.name == 'gin-rummy':
            from rlcard.games.gin_rummy.utils.action_event import ActionEvent
            try:
                return str(ActionEvent.decode_action(action_int))
            except:
                pass
        elif env.name == 'leduc-holdem':
            action_map = {0: 'call', 1: 'raise', 2: 'fold', 3: 'check'}
            if action_int in action_map:
                return action_map[action_int]
        elif env.name == 'limit-holdem':
            action_map = {0: 'call', 1: 'raise', 2: 'fold', 3: 'check'}
            if action_int in action_map:
                return action_map[action_int]
        elif env.name == 'no-limit-holdem':
            action_map = {0: 'fold', 1: 'check', 2: 'call', 3: 'raise'}
            if action_int in action_map:
                return action_map[action_int]
        else:
            # Fallback: try to use env's decode method if available
            try:
                return env._decode_action(action_int)
            except:
                pass
    
    # Fallback to string conversion
    return str(action)


def _extract_gin_rummy_obs(state, env):
    """
    Extract readable observation from Gin Rummy state.
    
    Gin Rummy raw_obs is a (5, 52) numpy array:
    - Row 0: Cards in hand (1 = has card)
    - Row 1: Top discard card
    - Row 2: Dead cards (discarded and known)
    - Row 3: Opponent known cards
    - Row 4: Unknown cards (in stock or opponent hand)
    
    Returns a dict with human-readable fields.
    """
    from rlcard.games.gin_rummy.utils.action_event import ActionEvent
    
    raw_obs = state.get('raw_obs', None)
    raw_legal_actions = state.get('raw_legal_actions', [])
    
    # Card mapping: 52 cards, 4 suits x 13 ranks
    # Index = suit * 13 + rank, where:
    # - Suits: 0=Spades, 1=Hearts, 2=Diamonds, 3=Clubs
    # - Ranks: 0=Ace, 1=2, ..., 12=King
    suits = ['S', 'H', 'D', 'C']
    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    
    def indices_to_cards(row):
        """Convert a binary row to list of card strings."""
        cards = []
        if isinstance(row, np.ndarray):
            row = row.tolist()
        for idx, val in enumerate(row):
            if val == 1:
                suit = suits[idx // 13]
                rank = ranks[idx % 13]
                cards.append(f"{rank}{suit}")
        return cards
    
    def get_top_discard(row):
        """Get the single top discard card."""
        cards = indices_to_cards(row)
        return cards[0] if cards else None
    
    # Extract info from numpy array
    if isinstance(raw_obs, np.ndarray) and raw_obs.shape == (5, 52):
        hand = indices_to_cards(raw_obs[0])
        top_discard = get_top_discard(raw_obs[1])
        dead_cards = indices_to_cards(raw_obs[2])
        opponent_known_cards = indices_to_cards(raw_obs[3])
        unknown_cards = indices_to_cards(raw_obs[4])
        stock_pile_num = len(unknown_cards)
    else:
        # Fallback for unexpected format
        hand = []
        top_discard = None
        dead_cards = []
        opponent_known_cards = []
        stock_pile_num = 0
    
    # Convert raw_legal_actions to readable format
    legal_actions = []
    for action in raw_legal_actions:
        if isinstance(action, int):
            try:
                action_event = ActionEvent.decode_action(action)
                legal_actions.append(str(action_event))
            except:
                legal_actions.append(str(action))
        else:
            legal_actions.append(str(action))
    
    return {
        'hand': hand,
        'top_discard': top_discard,
        'dead_cards': dead_cards,
        'opponent_known_cards': opponent_known_cards,
        'stock_pile_num': stock_pile_num,
        'legal_actions': legal_actions,
        'player_id': 0,  # Will be set by caller if needed
    }


def reorganize(trajectories, payoffs, env=None):
    ''' Reorganize the trajectory to make it RL friendly

    Args:
        trajectory (list): A list of trajectories
        payoffs (list): A list of payoffs for the players. Each entry corresponds to one player

    Returns:
        (list): A new trajectories that can be fed into RL algorithms.

    '''
    num_players = len(trajectories)
    new_trajectories = [[] for _ in range(num_players)]

    for player in range(num_players):
        for i in range(0, len(trajectories[player])-2, 2):
            if i == len(trajectories[player])-3:
                reward = payoffs[player]
                done = True
            else:
                reward, done = 0, False
            transition = trajectories[player][i:i+2].copy()
            transition.append(reward)
            transition.append(done)
            
            # Safe access to action_record (not all games have this field, e.g., gin-rummy)
            action_record = transition[0].get('action_record', [])
            transition.append(action_record[-20:] if action_record else [])
            
            # Handle different observation formats
            raw_obs = transition[0].get('raw_obs', {})
            
            # Check if raw_obs is a numpy array (gin-rummy) or dict (most other games)
            if isinstance(raw_obs, np.ndarray):
                # For gin-rummy and similar games with array observations
                if env is not None and env.name == 'gin-rummy':
                    obs_dict = _extract_gin_rummy_obs(transition[0], env)
                    obs_dict['player_id'] = player
                    transition[0] = obs_dict
                else:
                    # Generic fallback for array observations
                    transition[0] = {
                        'obs_array': raw_obs.tolist() if isinstance(raw_obs, np.ndarray) else raw_obs,
                        'legal_actions': transition[0].get('raw_legal_actions', []),
                        'player_id': player,
                    }
            else:
                # For games with dict observations (uno, leduc, etc.)
                transition[0] = raw_obs
            
            # Convert action to string using the helper function
            transition[1] = _convert_action_to_str(transition[1], env)
            
            if not isinstance(transition[2], float):
                transition[2] = float(transition[2])

            new_trajectories[player].append(transition)
    return new_trajectories

def evaluate(args):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    # Load models
    agents = []
    for position, model_path in enumerate(args.models):
        agent = load_model(model_path, env, position, device)
        # agent.use_raw = True
        agents.append(agent)
    env.set_agents(agents)

    # Evaluate
    rewards, all_trajectories, all_rewards = tournament(env, args.num_games)
    for position, reward in enumerate(rewards):
        reward = reward / args.num_games
        print(position, args.models[position], reward)

    if args.out_dir:
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        # Save trajectories
        for idx, (trajectory, reward) in enumerate(zip(all_trajectories, all_rewards)):
            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectory, reward, env)
            for position, model_path in enumerate(args.models):
                # print(trajectories[position])
                # with open(f"trajectories/{args.env}_{idx}_{position}_{args.models[position].split('/')[-1]}.txt", 'a') as f:
                #     f.write(json.dumps(trajectories[position]) + '\n')
                with open(f"{args.out_dir}/{time_str}-{args.env}_{position}_{args.models[position].split('/')[-1]}.txt", 'a') as f:
                    json_data = json.dumps(trajectories[position], cls=RLCardEncoder)
                    f.write(json_data + '\n')

def mp_simulate(q, args, worker_id):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    seed = args.seed+worker_id*100
    set_seed(seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': seed})

    # Load models
    agents = []
    for position, model_path in enumerate(args.models):
        agent = load_model(model_path, env, position, device)
        # agent.use_raw = True
        agents.append(agent)
    env.set_agents(agents)

    # Evaluate
    rewards, all_trajectories, all_rewards = tournament(env, args.num_games / args.num_workers)
    # for position, reward in enumerate(rewards):
    #     print(position, args.models[position], reward)

    if args.out_dir:
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        # Save trajectories
        for idx, (trajectory, reward) in enumerate(zip(all_trajectories, all_rewards)):
            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectory, reward, env)
            for position, model_path in enumerate(args.models):
                # print(trajectories[position])
                # with open(f"trajectories/{args.env}_{idx}_{position}_{args.models[position].split('/')[-1]}.txt", 'a') as f:
                #     f.write(json.dumps(trajectories[position]) + '\n')
                with open(f"{args.out_dir}/{time_str}-{worker_id}-{args.env}_{position}_{args.models[position].split('/')[-1]}.txt", 'a') as f:
                    json_data = json.dumps(trajectories[position], cls=RLCardEncoder)
                    f.write(json_data + '\n')
    # request data
    count_records = [[0, 0] for _ in range(len(args.models))]              
    for position, agent in enumerate(agents):
        if hasattr(agent, 'request_count'):
            count_records[position][0] = agent.request_count
            count_records[position][1] = agent.correct_count
    all_data = [rewards, count_records]
    q.put(all_data)

def evaluate_mp(args):
    for m in args.models:
        if m in model_config:
            args.num_workers = min(args.num_workers, model_config[m]['call_func']['num_workers'])

    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []
    for worker_id in range(args.num_workers):
        p = ctx.Process(
                target=mp_simulate,
                args=(q, args, worker_id))
        p.start()
        processes.append(p)

    all_rewards = [0 for _ in range(len(args.models))]
    all_count_records = [[0, 0] for _ in range(len(args.models))] 
    for p in processes:
        p.join()
    for i in range(args.num_workers):
        all_result = q.get()
        result = all_result[0]
        for j in range(len(result)):
            all_rewards[j] += result[j]
        result_count = all_result[1]
        for j in range(len(result_count)):
            all_count_records[j][0] += result_count[j][0]
            all_count_records[j][1] += result_count[j][1]
    print('Rewards results:')
    for position, reward in enumerate(all_rewards):
        reward = reward / args.num_games
        print(position, args.models[position], f'{reward:.3f}')
    print('Format Accuracy results:')
    for position, reward in enumerate(all_count_records):
        if reward[0] == 0:
            reward = -1
        else:
            reward = reward[1] / reward[0]
        print(position, args.models[position], f'{reward:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='leduc-holdem',
        choices=[
            'leduc-holdem',
            'limit-holdem',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
        ],
    )
    parser.add_argument(
        '--models',
        nargs='*',
        default=[
            'experiments/leduc_holdem_dqn_result/model.pth',
            'random',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_games',
        type=int,
        default=10000,
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=5,
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default=None,
    )
    args = parser.parse_args()
    print('Start evaluation at:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(args)

    if args.out_dir is not None and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    start = time.time()
    # evaluate(args)
    evaluate_mp(args)
    end = time.time()
    print('Time in minutes:', (end - start) / 60)