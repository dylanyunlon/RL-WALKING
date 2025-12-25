#!/bin/bash
# # # no-limit-holdem
# vs random
# python -m util.rlcard_util.gen_data --out_dir ./logs_eval/nolimit --env no-limit-holdem \
#     --models llm random \
#     --cuda 0  --num_games 100 --seed 44
# vs dqn
# model_path='/workspace/ww/rl-card-results-1226/no-limit-holdem/checkpoint_dqn.pt'
# python -m util.rlcard_util.gen_data --out_dir ./logs_eval/nolimit --env no-limit-holdem \
#     --models llm $model_path \
#     --cuda 0 --num_games 100 --seed 44

python -m util.rlcard_util.gen_data --out_dir $6 --env no-limit-holdem \
    --models $4 $5 \
    --cuda $3 --num_games $2 --seed $1