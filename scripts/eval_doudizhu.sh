# # douzizhu
# python util/douzero_util/generate_eval_data.py --num_games 100 --output ../DouZero/test_douzizhu44 --seed 44
# vs pre
# python eval_llm_douzero.py --eval_data ../DouZero/test_douzizhu44.pkl --log_dir ./logs_eval/dou \
#   --landlord llm \
#   --landlord_up ../DouZero/baselines/douzero_WP/landlord_up.ckpt \
#   --landlord_down ../DouZero/baselines/douzero_WP/landlord_down.ckpt --gpu_device 0
# vs rule
# python eval_llm_douzero.py --eval_data ../DouZero/test_douzizhu44.pkl --log_dir ./logs_eval/dou \
#   --landlord llm --landlord_up rlcard --landlord_down rlcard --gpu_device 0
# vs random
# python eval_llm_douzero.py --eval_data ../DouZero/test_douzizhu44.pkl --log_dir ./logs_eval/dou \
#   --landlord llm --landlord_up random --landlord_down random --gpu_device 0
# different roles
# vs pre
# python eval_llm_douzero.py --eval_data ../DouZero/test_douzizhu44.pkl --log_dir ./logs_eval/dou \
#   --landlord ../DouZero/baselines/douzero_WP/landlord.ckpt \
#   --landlord_up llm \
#   --landlord_down llm --gpu_device 0
# vs rule
# python eval_llm_douzero.py --eval_data ../DouZero/test_douzizhu44.pkl --log_dir ./logs_eval/dou \
#   --landlord rlcard --landlord_up llm --landlord_down llm --gpu_device 0
# vs random
# python eval_llm_douzero.py --eval_data ../DouZero/test_douzizhu44.pkl --log_dir ./logs_eval/dou \
#   --landlord random --landlord_up llm --landlord_down llm --gpu_device 0

# # douzizhu
python util/douzero_util/generate_eval_data.py --num_games $2 --output ../DouZero/eval_s$1-$2 --seed $1
python eval_llm_douzero.py --eval_data ../DouZero/eval_s$1-$2.pkl --log_dir $6 \
  --landlord $4 \
  --landlord_up $5 \
  --landlord_down $5 --gpu_device $3
# python eval_llm_douzero.py --eval_data ../DouZero/eval_s$1-$2.pkl --log_dir $6 \
#   --landlord $5 \
#   --landlord_up $4 \
#   --landlord_down $4 --gpu_device $3
