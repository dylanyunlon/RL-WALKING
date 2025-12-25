#!/bin/bash
# riichi
# python -u -m util.mortal_util.one_vs_three --out_dir ./logs_eval/riichi \
#   --models llm moral \
#   --cuda 0  --num_games 100 --seed 44

python -u -m util.mortal_util.one_vs_three --out_dir $6 \
    --models $4 $5 \
    --cuda $3 --num_games $2 --seed $1