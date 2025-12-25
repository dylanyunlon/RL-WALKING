#!/bin/bash

# std=""
# std="_std"
std="_std2"
# final, ckpt, final_std, ckpt_std, fix
type="final$std"

# doudizhu4-1000000, guandan4-1000000, riichi4-1000000, uno4-400000, gin4-400000, leduc4-400000, limit4-400000, nolimit4-400000
# doudizhu4-guandan4-riichi4-uno4-limit4-nolimit4-leduc4-gin4-3100000
trained_model=doudizhu4-guandan4-riichi4-uno4-limit4-nolimit4-leduc4-gin4-3100000
base=qwen
game="doudizhu$std"
gpu=0

# final ckpt
bash deploy_eval_core.sh $trained_model $base $game $gpu $type