#!/bin/bash
cd ../Danzero_plus/wintest/torch
save_file=../logs_gen_6000
out_file="$save_file-out"
num_games=6000

# save_file=../logs_eval_std2
# out_file="$save_file-out"
# num_games=20

nohup ./danserver $num_games > /dev/null  2>&1 &
sleep 2s
nohup python -u ../danzero/client0.py --log_dir $save_file > $out_file-0 2>&1 &
sleep 5s
nohup python ../ai4/client2.py > /dev/null 2>&1 &
sleep 2s
nohup python -u ../danzero/client2.py --log_dir $save_file > $out_file-2 2>&1 &
sleep 5s
nohup python ../ai4/client4.py > /dev/null 2>&1 &
sleep 2s
nohup python ../danzero/actor.py > /dev/null 2>&1 &
