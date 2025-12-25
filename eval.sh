#!/bin/bash

evaluate_doudizhu() {
    local seed=$1
    local num_games=$2
    local cuda=$3
    local player1=$4
    local player2=$5
    local test_model=$6
    
    local out_dir="./${base_dir}/doudizhu-$seed-$num_games-$player1-$player2-$test_model"
    local eval_file="$out_dir-log.txt"
    bash scripts/eval_doudizhu.sh $seed $num_games $cuda $player1 $player2 $out_dir | tee -a $eval_file
}

evaluate_guandan() {
    local seed=$1
    local num_games=$2
    local cuda=$3
    local player1=$4
    local player2=$5
    local test_model=$6
    
    local out_dir="./${base_dir}/guandan-$seed-$num_games-$player1-$player2-$test_model"
    local eval_file="$out_dir-log.txt"
    bash scripts/eval_guandan.sh $seed $num_games $cuda $player1 $player2 $out_dir $eval_file
}

evaluate_riichi() {
    local seed=$1
    local num_games=$2
    local cuda=$3
    local player1=$4
    local player2=$5
    local test_model=$6
    
    local out_dir="./${base_dir}/riichi-$seed-$num_games-$player1-$player2-$test_model"
    local eval_file="$out_dir-log.txt"
    bash scripts/eval_riichi.sh $seed $num_games $cuda $player1 $player2 $out_dir | tee -a $eval_file
}

evaluate_uno() {
    local seed=$1
    local num_games=$2
    local cuda=$3
    local player1=$4
    local player2=$5
    local test_model=$6

    local out_dir="./${base_dir}/uno-$seed-$num_games-$player1-$player2-$test_model"
    local eval_file="$out_dir-log.txt"
    bash scripts/eval_uno.sh $seed $num_games $cuda $player1 $player2 $out_dir | tee -a $eval_file
}

evaluate_limit() {
    local seed=$1
    local num_games=$2
    local cuda=$3
    local player1=$4
    local player2=$5
    local test_model=$6
    
    local out_dir="./${base_dir}/limit-$seed-$num_games-$player1-$player2-$test_model"
    local eval_file="$out_dir-log.txt"
    bash scripts/eval_limit.sh $seed $num_games $cuda $player1 $player2 $out_dir | tee -a $eval_file
}

evaluate_leduc() {
    local seed=$1
    local num_games=$2
    local cuda=$3
    local player1=$4
    local player2=$5
    local test_model=$6
    
    local out_dir="./${base_dir}/leduc-$seed-$num_games-$player1-$player2-$test_model"
    local eval_file="$out_dir-log.txt"
    bash scripts/eval_leduc.sh $seed $num_games $cuda $player1 $player2 $out_dir | tee -a $eval_file
}

evaluate_nolimit() {
    local seed=$1
    local num_games=$2
    local cuda=$3
    local player1=$4
    local player2=$5
    local test_model=$6
    
    local out_dir="./${base_dir}/nolimit-$seed-$num_games-$player1-$player2-$test_model"
    local eval_file="$out_dir-log.txt"
    bash scripts/eval_nolimit.sh $seed $num_games $cuda $player1 $player2 $out_dir | tee -a $eval_file
}

evaluate_gin() {
    local seed=$1
    local num_games=$2
    local cuda=$3
    local player1=$4
    local player2=$5
    local test_model=$6
    
    local out_dir="./${base_dir}/gin-$seed-$num_games-$player1-$player2-$test_model"
    local eval_file="$out_dir-log.txt"
    bash scripts/eval_gin.sh $seed $num_games $cuda $player1 $player2 $out_dir | tee -a $eval_file
}

# check and create log dir
check_dir() {
    local log_dir=$1
    if [ ! -d $log_dir ]; then
        mkdir -p $log_dir
    fi
}

test_game=$1
seed=$2
num_games=$3
cuda=$4
player1=$5
player2=$6
test_model=$7
base_dir=$8

# log dir
# base_dir='logs_eval_llm'
# base_dir='logs_eval'

# # Default parameters
# seed=44
# num_games=100
# cuda=0
# player1=llm

check_dir $base_dir
# run evaluation according to the test_game
if [ $test_game = "doudizhu" ]; then
    evaluate_doudizhu $seed $num_games $cuda $player1 $player2 $test_model
elif [ $test_game = "guandan" ]; then    
    evaluate_guandan $seed $num_games $cuda $player1 $player2 $test_model
elif [ $test_game = "riichi" ]; then
    evaluate_riichi $seed $num_games $cuda $player1 $player2 $test_model
elif [ $test_game = "uno" ]; then
    evaluate_uno $seed $num_games $cuda $player1 $player2 $test_model
elif [ $test_game = "limit" ]; then
    evaluate_limit $seed $num_games $cuda $player1 $player2 $test_model
elif [ $test_game = "leduc" ]; then
    evaluate_leduc $seed $num_games $cuda $player1 $player2 $test_model
elif [ $test_game = "nolimit" ]; then
    evaluate_nolimit $seed $num_games $cuda $player1 $player2 $test_model
elif [ $test_game = "gin" ]; then
    evaluate_gin $seed $num_games $cuda $player1 $player2 $test_model
else
    echo "Invalid game"
fi