import os 
import argparse
import time
from datetime import datetime

from rlcard.utils import set_seed
from util.douzero_util.simulation import evaluate
from util.douzero_util.simulation import generate_training_data as evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    'Dou Dizhu Evaluation')
    parser.add_argument('--landlord', type=str,
            default='baselines/douzero_ADP/landlord.ckpt')
    parser.add_argument('--landlord_up', type=str,
            default='baselines/sl/landlord_up.ckpt')
    parser.add_argument('--landlord_down', type=str,
    
            default='baselines/sl/landlord_down.ckpt')
    parser.add_argument('--eval_data', type=str,
            default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--gpu_device', type=str, default='')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    args = parser.parse_args()
    print('Start evaluation at:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(args)
    
    # Seed numpy, torch, random
    set_seed(args.seed)

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    if args.log_dir is not None and not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    start = time.time()
    evaluate(args.landlord,
             args.landlord_up,
             args.landlord_down,
             args.eval_data,
             args.num_workers,
             args.seed,
             args.log_dir)
    end = time.time()
    print('Time in minutes:', (end - start) / 60)
