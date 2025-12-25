#!/bin/bash

# std=""
# std="_std"
std="_std2"

test_model=api
base_dir="logs_eval_llm_api$std"
cuda=1
game="riichi$std"

# game info
game_name="game_$game"
game_info_file="eval_conf/game_info.conf"
source $game_info_file
test_game=$(eval echo \${${game_name}[0]})
seed=$(eval echo \${${game_name}[1]})
num_games=$(eval echo \${${game_name}[2]})
# player1=$(eval echo \${${game_name}[3]})
player2=$(eval echo \${${game_name}[4]})
echo $test_game

llms=('glm4-air' 'glm4-plus' 'gpt4o' 'gpt4om')
for player1 in "${llms[@]}"; do
    bash eval.sh $test_game $seed $num_games $cuda $player1 $player2 $test_model $base_dir
    if [ "$test_game" == "doudizhu" ]; then
        bash eval.sh $test_game $seed $num_games $cuda $player2 $player1 $test_model $base_dir
    fi
done

