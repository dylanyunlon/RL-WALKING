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

class RLCardEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, MahjongCard):
            return obj.get_str()
        if isinstance(obj, Action):
            return obj.name
        if isinstance(obj, Stage):
            return obj.name
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
        action: The action to convert (can be int, str, or other types)
        env: The RLCard environment
    
    Returns:
        str: String representation of the action
    """
    # Already a string, return as-is
    if isinstance(action, str):
        return action
    
    # MahjongCard type
    if isinstance(action, MahjongCard):
        return action.get_str()
    
    # Integer action ID - need to convert based on game type
    if isinstance(action, int):
        if env.name == 'uno':
            from rlcard.games.uno.utils import ACTION_LIST
            return ACTION_LIST[action]
        elif env.name == 'gin-rummy':
            from rlcard.games.gin_rummy.utils.action_event import ActionEvent
            return ActionEvent.decode_action(action)
        elif env.name == 'leduc-holdem':
            action_map = {0: 'call', 1: 'raise', 2: 'fold', 3: 'check'}
            return action_map.get(action, str(action))
        elif env.name == 'limit-holdem':
            action_map = {0: 'call', 1: 'raise', 2: 'fold', 3: 'check'}
            return action_map.get(action, str(action))
        elif env.name == 'no-limit-holdem':
            # No-limit holdem has more complex action space
            action_map = {0: 'fold', 1: 'check', 2: 'call', 3: 'raise'}
            return action_map.get(action, str(action))
        else:
            # Fallback: try to use env's decode method if available
            try:
                return env._decode_action(action)
            except:
                return str(action)
    
    # For other types (like Action enum from nolimitholdem)
    if isinstance(action, Action):
        return action.name
    
    # Fallback to string conversion
    return str(action)


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
            if i ==len(trajectories[player])-3:
                reward = payoffs[player]
                done =True
            else:
                reward, done = 0, False
            transition = trajectories[player][i:i+2].copy()
            transition.append(reward)
            transition.append(done)
            transition.append(transition[0]['action_record'][-20:])
            transition[0] = transition[0]['raw_obs']
            
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