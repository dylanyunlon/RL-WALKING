# douzizhu
python util/douzero_util/generate_eval_data.py --num_games 100000 --output ../DouZero/eval_v3 --seed 43
python eval_llm_douzero.py --eval_data ../DouZero/eval_v3.pkl --log_dir ./data_gen/douzero31 \
  --landlord ../DouZero/baselines/douzero_WP/landlord.ckpt \
  --landlord_up rlcard \
  --landlord_down rlcard --gpu_device 0
python eval_llm_douzero.py --eval_data ../DouZero/eval_v3.pkl --log_dir ./data_gen/douzero31 \
  --landlord rlcard \
  --landlord_up ../DouZero/baselines/douzero_WP/landlord_up.ckpt \
  --landlord_down ../DouZero/baselines/douzero_WP/landlord_down.ckpt --gpu_device 0

# guandan
bash scripts/gen_data_guandan.sh

# riichi
bash scripts/gen_data_riichi.sh

# rlcard
# uno
python -m util.rlcard_util.gen_data --out_dir ./data_gen/rlcard_uno-rulevr-s45-50000 --env uno \
  --models uno-rule-v1 random --cuda 0  --num_games 50000 --seed 45

# gin-rummy
python -m util.rlcard_util.gen_data --out_dir ./data_gen/rlcard_gin-rulevr-s45-50000 --env gin-rummy \
  --models gin-rummy-novice-rule random --cuda 0  --num_games 50000 --seed 45

# leduc-holdem
model_path='/workspace/ww/rl-card-results-1226/leduc-holdem/checkpoint_dqn.pt'
python -m util.rlcard_util.gen_data --out_dir ./data_gen/rlcard_leduc-dqn-s45-400000 --env leduc-holdem \
  --models $model_path random --cuda 0  --num_games 400000 --seed 45

limit-holdem
model_path='/workspace/ww/rl-card-results-1226/limit-holdem/checkpoint_dqn.pt'
python -m util.rlcard_util.gen_data --out_dir ./data_gen/rlcard_limit-dqn-s45-200000  --env limit-holdem \
  --models $model_path random --cuda 1  --num_games 200000 --seed 45

# nolimit-holdem
model_path='/workspace/ww/rl-card-results-1226/no-limit-holdem/checkpoint_dqn.pt'
python  -m util.rlcard_util.gen_data --out_dir ./data_gen/rlcard_nolimit-dqn-s45-400000 --env no-limit-holdem \
  --models $model_path random --cuda 0  --num_games 400000 --seed 45