#!/bin/bash
# uno
# vs rule
# python -m util.rlcard_util.gen_data --out_dir ./logs_eval/uno --env uno \
#   --models llm uno-rule-v1 \
#   --cuda 0  --num_games 100 --seed 44
# vs random
# python -m util.rlcard_util.gen_data --out_dir ./logs_eval/uno --env uno \
#   --models llm random \
#   --cuda 0  --num_games 100 --seed 44

python -m util.rlcard_util.gen_data --out_dir $6 --env uno \
    --models $4 $5 \
    --cuda $3 --num_games $2 --seed $1