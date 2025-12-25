#!/bin/bash

data_config="data/data_mix.json"
target_config="data/dataset_info.json"
cp $data_config $target_config

max_examples_values=(3100000)
datasets=(doudizhu4,guandan4,riichi4,uno4,limit4,nolimit4,leduc4,gin4)
# datasets=(uno4 limit4 nolimit4 leduc4 gin4)

source_config="train_config/card_lora_sft.yaml"
temp_config="train_config/card_lora_sft_temp.yaml"

PRE_MODEL_DIR="Qwen25-14B-Instruct"
OUT_BASE_DIR="/workspace/ww/output/card_qwen"
template=qwen

# PRE_MODEL_DIR="Meta-Llama-3_1-8B-Instruct"
# OUT_BASE_DIR="/workspace/ww/output/card"
# template=llama3

# PRE_MODEL_DIR="glm-4-9b-chat"
# OUT_BASE_DIR="/workspace/ww/output/card"
# template=glm4

# iterate over each dataset and each max_examples value
for dataset in "${datasets[@]}"; do
    for max_examples in "${max_examples_values[@]}"; do
        echo "Running training with dataset=${dataset} and max_examples=${max_examples}"
        
        cp $source_config $temp_config
        # use sed to replace the max_examples and dataset values in the YAML file
        sed -i "s/^max_samples: .*/max_samples: ${max_examples}/"  $temp_config
        sed -i "s/^dataset: .*/dataset: ${dataset}/"  $temp_config
        sed -i "s|^output_dir: .*|output_dir: $OUT_BASE_DIR-${PRE_MODEL_DIR}/$(echo ${dataset} | tr ',' '-')-${max_examples}|"  $temp_config
        sed -i "s|^model_name_or_path: .*|model_name_or_path: /workspace/ww/pretrained_models/${PRE_MODEL_DIR}|"  $temp_config
        sed -i "s|^template: .*|template: ${template}|" $temp_config

        llamafactory-cli train  $temp_config
    done
done