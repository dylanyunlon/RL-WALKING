#!/bin/bash
# # # limit-holdem
# vs rule
# python -m util.rlcard_util.gen_data --out_dir ./logs_eval/limit --env limit-holdem \
#     --models llm limit-holdem-rule-v1 \
#     --cuda 0  --num_games 100 --seed 44
# vs random
# python -m util.rlcard_util.gen_data --out_dir ./logs_eval/limit --env limit-holdem \
#     --models llm random \
#     --cuda 0  --num_games 100 --seed 44
# vs dqn
# model_path='/workspace/ww/rl-card-results-1226/limit-holdem/checkpoint_dqn.pt'
# python -m util.rlcard_util.gen_data --out_dir ./logs_eval/limit --env limit-holdem \
#     --models llm $model_path \
#     --cuda 0 --num_games 100 --seed 44

python -m util.rlcard_util.gen_data --out_dir $6 --env limit-holdem \
    --models $4 $5 \
    --cuda $3 --num_games $2 --seed $1