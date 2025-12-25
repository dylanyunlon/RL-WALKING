#!/bin/bash
# # leduc-holdem
# vs cfr
# python -m util.rlcard_util.gen_data --out_dir ./logs_eval/leduc --env leduc-holdem \
#     --models llm /workspace/ww/rlcard-master/rlcard/models/pretrained/leduc_holdem_cfr \
#     --cuda 0 --num_games 100 --seed 44
# vs rule
# python -m util.rlcard_util.gen_data --out_dir ./logs_eval/leduc --env leduc-holdem \
#     --models llm leduc-holdem-rule-v2 \
#     --cuda 0 --num_games 100 --seed 44
# vs random
# python -m util.rlcard_util.gen_data --out_dir ./logs_eval/leduc --env leduc-holdem \
#     --models llm random \
#     --cuda 0 --num_games 100 --seed 44
# vs dqn
# model_path='/workspace/ww/rl-card-results-1226/leduc-holdem/checkpoint_dqn.pt'
# python -m util.rlcard_util.gen_data --out_dir ./logs_eval/leduc --env leduc-holdem \
#     --models llm $model_path \
#     --cuda 0 --num_games 100 --seed 44


python -m util.rlcard_util.gen_data --out_dir $6 --env leduc-holdem \
    --models $4 $5 \
    --cuda $3 --num_games $2 --seed $1