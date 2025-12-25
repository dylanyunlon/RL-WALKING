#!/bin/bash
PRE_BASE_DIR="/workspace/ww/pretrained_models"

# model info
trained_model="$1"
base="$2"
game="$3"
gpu="$4"

# final, ckpt, pre
type="$5"
sub_type=${type:0:3}

base_name="base_$base"
game_name="game_$game"
gpu_name="gpu_$gpu"

# config file
base_model_file="eval_conf/base_model.conf"
game_info_file="eval_conf/game_info.conf"
gpu_info_file="eval_conf/gpu_info.conf"
# import predefined config
source $base_model_file
source $game_info_file
source $gpu_info_file

# base model info
PRE_MODEL_DIR=$(eval echo \${${base_name}[0]})
OUT_BASE_DIR=$(eval echo \${${base_name}[1]})
source_config=$(eval echo \${${base_name}[2]})
temp_config=$(eval echo \${${base_name}[3]})
echo $PRE_MODEL_DIR

# gpu info
api_port=$(eval echo \${${gpu_name}[0]})
gpu_server=$(eval echo \${${gpu_name}[1]})
gpu_client=$(eval echo \${${gpu_name}[2]})
echo $api_port

start_server() {
    local log_file=$1
    
    cp $source_config $temp_config
    sed -i "s|^model_name_or_path: .*|model_name_or_path: $PRE_BASE_DIR/${PRE_MODEL_DIR}|" $temp_config
    sed -i "s|^adapter_name_or_path: .*|adapter_name_or_path: $OUT_BASE_DIR-${PRE_MODEL_DIR}/${trained_model}|" $temp_config

    if [ "$sub_type" = "pre" ]; then
        sed -i "s|^adapter_name_or_path: .*|# adapter_name_or_path: |" $temp_config
    fi

    export API_PORT=$api_port
    export CUDA_VISIBLE_DEVICES=$gpu_server
    nohup llamafactory-cli api $temp_config > $log_file 2>&1 &
    process_id=$!

    echo $process_id
}

run_eval() {
    export API_PORT=$api_port
    cuda=$gpu_client
    test_model="$(echo ${trained_model} | tr '/' '-')-t${API_TEMP}"
    base_dir=logs_eval_$type-$PRE_MODEL_DIR

    # game info
    test_game=$(eval echo \${${game_name}[0]})
    seed=$(eval echo \${${game_name}[1]})
    num_games=$(eval echo \${${game_name}[2]})
    player1=$(eval echo \${${game_name}[3]})
    player2=$(eval echo \${${game_name}[4]})
    echo $test_game
    
    bash eval.sh $test_game $seed $num_games $cuda $player1 $player2 $test_model $base_dir
    if [ "$test_game" == "doudizhu" ]; then
        bash eval.sh $test_game $seed $num_games $cuda $player2 $player1 $test_model $base_dir
    fi
}

check_server() {
    local log_file=$1
    local max_attempts=60
    local attempt=1

    # wait for server to start
    while [ $attempt -le $max_attempts ]; do
        sleep 30s
        
        # check if server started successfully by checking deploy_out.log
        if grep -q "running on http://0.0.0.0:${api_port} (Press CTRL+C to quit)" $log_file; then
            echo "Server started successfully. Now running eval.sh..."
            break
        else
            echo "Attempt $attempt/$max_attempts: Server did not start successfully."
        fi
        
        attempt=$((attempt + 1))
    done

    if [ $attempt -gt $max_attempts ]; then
        echo "Server failed to start after $max_attempts attempts."
        return 1
    fi

    return 0
}

# main function
log_file="./deploy_out_fix/deploy_out-$PRE_MODEL_DIR-$(echo ${trained_model} | tr '/' '-')-${game}.log"
echo $log_file

process_id=$(start_server $log_file)
if check_server $log_file; then
    run_eval
fi

kill $process_id
