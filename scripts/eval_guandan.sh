#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH


# seed=$1
num_games=$2
# cuda=$3
player1=$4
# player2=$5
out_dir=$6
eval_file=$7

# start
GUAN_HOME=/workspace/ww/project/card_agent/Danzero_plus/wintest
# nohup $GUAN_HOME/torch/danserver 20 > /dev/null  2>&1 &
nohup $GUAN_HOME/torch/danserver $num_games > /dev/null  2>&1 &
process_id1=$!
sleep 2s
nohup python -u -m util.guandan_util.client0 --log_dir $out_dir >> $eval_file-0 2>&1 &
process_id2=$!
sleep 2s
nohup python $GUAN_HOME/ai4/client2.py > /dev/null 2>&1 &
process_id3=$!
sleep 2s
nohup python -u -m util.guandan_util.client2 --log_dir $out_dir >> $eval_file-2 2>&1 &
process_id4=$!
sleep 2s
nohup python $GUAN_HOME/ai4/client4.py > /dev/null 2>&1 &
process_id5=$!
sleep 2s
nohup python -m util.guandan_util.actor_llm --model $player1 >> $eval_file-0 2>&1 &
process_id6=$!
# python -m util.guandan_util.actor

# 定义最大重试次数
max_attempts=6000
attempt=1

# 等待服务器启动
while [ $attempt -le $max_attempts ]; do
    sleep 60s
    
    # 检查 deploy_out.log 中是否包含成功启动的关键字
    last_line=$(tail -n 1 "$eval_file-0")
    if echo "$last_line" | grep -q "对局结束"; then
        echo "Eval finish. Now stop server ..."
        break
    else
        echo "Attempt $attempt/$max_attempts: Eval did not finish."
    fi
    
    attempt=$((attempt + 1))
done

# stop
kill $process_id1
kill $process_id2
kill $process_id3
kill $process_id4
kill $process_id5
kill $process_id6
ps -ef | grep actor_llm | awk '{print $2}' | xargs kill -9

