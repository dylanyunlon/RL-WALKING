#!/bin/bash

# std=""
# std="_std"
std="_std2"
# final, ckpt, final_std, ckpt_std, pre
type="pre$std"

base=glm
trained_model=$base
# game="doudizhu$std"
gpu=6

games=(doudizhu riichi uno gin leduc limit nolimit)
for game_temp in "${games[@]}"; do
    game="$game_temp$std"
    # final ckpt
    bash deploy_eval_core.sh $trained_model $base $game $gpu $type
done