CUDA_VISIBLE_DEVICES=0 python ./util/length_cdf.py \
  --model_name_or_path '/workspace/ww/pretrained_models/glm-4-9b-chat' \
  --dataset_dir '/workspace/ww/project/card_agent/llm4cardgame/data' \
  --dataset guandan3 \
  --template glm4
