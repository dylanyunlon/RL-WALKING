#!/bin/bash

# std=""
# std="_std"
std="_std2"
# final, ckpt, final_std, ckpt_std
type="ckpt$std"

# doudizhu4-1000000, guandan4-1000000, riichi4-1000000, uno4-400000, gin4-400000, leduc4-400000, limit4-400000, nolimit4-400000
# doudizhu4-guandan4-riichi4-uno4-limit4-nolimit4-leduc4-gin4-3100000
trained_model=doudizhu4-guandan4-riichi4-uno4-limit4-nolimit4-leduc4-gin4-3100000
base=glm
game="doudizhu$std"
gpu=4

# base model info
base_model_file="eval_conf/base_model.conf"
source $base_model_file
base_name="base_$base"
PRE_MODEL_DIR=$(eval echo \${${base_name}[0]})
OUT_BASE_DIR=$(eval echo \${${base_name}[1]})

# all ckpt
CHECKPOINT_DIRS=$(find "$OUT_BASE_DIR-${PRE_MODEL_DIR}/$trained_model" -type d -name "*checkpoint*" -exec basename {} \;)
for temp in $CHECKPOINT_DIRS; do
# for temp in "${CHECKPOINT_DIRS[@]}"; do
    echo "$trained_model/$temp"
    bash deploy_eval_core.sh $trained_model/$temp $base $game $gpu $type
done