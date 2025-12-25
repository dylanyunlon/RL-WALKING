#!/bin/bash
cd util/mahjong_util
python merge_data.py --input_dir '/workspace/ww/data/mahjong_data/' --output_dir './es4p.db'
python gen_mjai_data.py --input_dir './es4p.db' --output_dir './logs' --num_games 7000
python gen_action_data.py --input_dir './logs' --output_dir './logs_format4'