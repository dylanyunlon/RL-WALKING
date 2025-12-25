
import numpy as np
import torch
import secrets
import os
from util.mortal_util.model import Brain, DQN
from util.mortal_util.engine import MortalEngine
from util.mahjong_util.libriichi.arena import OneVsThree
from util.mortal_util.config import config
from util.mortal_util.llm_engine import LLMEngine
import time
from datetime import datetime
import argparse
from rlcard.utils import set_seed

def load_mortal_engine(config_mortal):
    state = torch.load(config_mortal['state_file'], map_location=torch.device('cpu'))
    cham_cfg = state['config']
    version = cham_cfg['control'].get('version', 1)
    conv_channels = cham_cfg['resnet']['conv_channels']
    num_blocks = cham_cfg['resnet']['num_blocks']
    mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])
    if config_mortal['enable_compile']:
        mortal.compile()
        dqn.compile()
    engine_cham = MortalEngine(
        mortal,
        dqn,
        is_oracle = False,
        version = version,
        device = torch.device(config_mortal['device']),
        enable_amp = config_mortal['enable_amp'],
        enable_rule_based_agari_guard = config_mortal['enable_rule_based_agari_guard'],
        name = config_mortal['name'],
    )
    return engine_cham

def main(args):

    cfg = config['1v3']
    games_per_iter = cfg['games_per_iter']
    seeds_per_iter = games_per_iter // 4
    iters = cfg['iters']
    log_dir = cfg['log_dir']
    key = cfg.get('seed_key', -1)

    if args.seed is not None:
        key = args.seed
    if args.num_games is not None:
        games_per_iter = args.num_games
        seeds_per_iter = games_per_iter // 4
    if args.out_dir is not None:
        log_dir = args.out_dir

    if key == -1:
        key = secrets.randbits(64)

    if args.models[0] == 'mortal':
        engine_chal = load_mortal_engine(cfg['challenger'])
    else:
        engine_chal = LLMEngine(4, model=args.models[0])
    if args.models[1] == 'mortal':
        engine_cham = load_mortal_engine(cfg['champion'])
    else:
        engine_chal = LLMEngine(4, model=args.models[1])

    seed_start = 10000
    for i, seed in enumerate(range(seed_start, seed_start + seeds_per_iter * iters, seeds_per_iter)):
        print('-' * 50)
        print('#', i)
        env = OneVsThree(
            disable_progress_bar = True,
            log_dir = log_dir,
        )
        rankings = env.py_vs_py(
            challenger = engine_chal,
            champion = engine_cham,
            seed_start = (seed, key),
            seed_count = seeds_per_iter,
        )
        rankings = np.array(rankings)
        first_rank = rankings[0] / rankings.sum()
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        avg_pt = rankings @ np.array([90, 45, 0, -135]) / rankings.sum()
        print(args.models)
        print(f'challenger rankings: {rankings} ({first_rank:.3f}, {avg_rank}, {avg_pt}pt)')
    
    print('Format Accuracy results:')
    for pos, engine in enumerate([engine_chal, engine_cham]):
        if hasattr(engine, 'request_count') and engine.request_count > 0:
            format_acc = engine.correct_count / engine.request_count
        else:
            format_acc = -1
        print(f"{pos} {args.models[pos]} {format_acc:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Riichi Evaluation')
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--models',
        nargs='*',
        default=[
            'llm',
            'mortal',
        ],
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--num_games',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default=None,
    )
    args = parser.parse_args()
    print('Start evaluation at:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(args)

    # Seed numpy, torch, random
    set_seed(args.seed)

    if args.out_dir is not None and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    try:
        start = time.time()
        main(args)
        end = time.time()
        print('Time in minutes:', (end - start) / 60)
    except KeyboardInterrupt:
        pass
