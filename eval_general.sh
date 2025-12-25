#!/bin/bash
PRE_BASE_DIR="/workspace/ww/pretrained_models"

# doudizhu4-1000000, guandan4-1000000, riichi4-1000000, uno4-400000, gin4-400000, leduc4-400000, limit4-400000, nolimit4-400000
# doudizhu4-guandan4-riichi4-uno4-limit4-nolimit4-leduc4-gin4-3100000
FT_MODEL_DIR="doudizhu4-1000000/checkpoint-5600"

# qwen, llama, glm
base=glm

# base model info
os_eval_file="eval_conf/os_eval.conf"
source $os_eval_file
base_name="base_$base"
PRE_MODEL_DIR=$(eval echo \${${base_name}[0]})
OUT_BASE_DIR=$(eval echo \${${base_name}[1]})
source_config=$(eval echo \${${base_name}[2]})
echo $PRE_MODEL_DIR

temp_config="eval_config/empty_model.py"
cp $source_config $temp_config
new_abbr="${PRE_MODEL_DIR}-$(echo ${FT_MODEL_DIR} | tr '/' '-')"
new_path="${PRE_BASE_DIR}/${PRE_MODEL_DIR}"
new_peft_path="${OUT_BASE_DIR}-${PRE_MODEL_DIR}/${FT_MODEL_DIR}"


# replace abbr, path, peft_path
sed -i "s#\([[:space:]]*abbr=\)[^,]*#\1'${new_abbr}'#" $temp_config
sed -i "s#\([[:space:]]*path=\)[^,]*#\1'${new_path}'#" $temp_config
sed -i "s#\([[:space:]]*peft_path=\)[^,]*#\1'${new_peft_path}'#" $temp_config

export CUDA_VISIBLE_DEVICES=5,6
cd eval_config
opencompass ./eval_general_model.py -w outputs/$new_abbr