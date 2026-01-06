#!/bin/bash
# ===========================================
# LLM4CardGame - Complete Training & Evaluation Pipeline (v5)
# ===========================================
# 
# NeurIPS 2025 Paper: "Can Large Language Models Master Complex Card Games?"
# Paper: https://openreview.net/forum?id=cmN8Wbvanr
# Original Repo: https://github.com/THUDM/LLM4CardGame
#
# v5 Changes:
# - Full RLCard game support (uno, gin, leduc, limit, nolimit)
# - Fixed _generate_rlcard_data() with correct parameters
# - Added RLCard teacher model paths
# - Added per-game training and evaluation pipelines
# - Added rlcard_full pipeline command
# - Kept doudizhu unchanged (already working with 50k data)
# ===========================================

set -e

export PATH="/data/jiacheng/anaconda3/envs/llm4cardgame_1/bin:$PATH"

# ===========================================
# Configuration - All Paths Relative to SCRIPT_DIR
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
TRAIN_CONFIG_DIR="${TRAIN_CONFIG_DIR:-$PROJECT_DIR/train_config}"

# HuggingFace cache directory (for downloaded models)
HF_CACHE_DIR="${HF_HOME:-/data/jiacheng/system/cache/temp/huggingface}"
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"

LOCAL_MODEL_CACHE="${LOCAL_MODEL_CACHE:-$SCRIPT_DIR/models}"

# ===========================================
# Supported Models (HuggingFace Hub Paths)
# ===========================================

declare -A MODEL_HF_PATHS=(
    ["qwen"]="Qwen/Qwen2.5-7B-Instruct"
    ["qwen14b"]="Qwen/Qwen2.5-14B-Instruct"
    ["glm4"]="THUDM/glm-4-9b-chat"
    ["llama3"]="meta-llama/Llama-3.1-8B-Instruct"
)

declare -A MODEL_TEMPLATES=(
    ["qwen"]="qwen"
    ["qwen14b"]="qwen"
    ["glm4"]="glm4"
    ["llama3"]="llama3"
)

# Default model selection
MODEL_KEY="${MODEL_KEY:-qwen}"
BASE_MODEL="${BASE_MODEL:-${MODEL_HF_PATHS[$MODEL_KEY]}}"
TEMPLATE="${TEMPLATE:-${MODEL_TEMPLATES[$MODEL_KEY]}}"

LORA_PATH="${LORA_PATH:-$OUTPUT_DIR/sft_lora}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
API_PORT="${API_PORT:-8555}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-llm4cardgame_1}"

# DouZero and DanZero directories
DOUZERO_DIR="${DOUZERO_DIR:-$SCRIPT_DIR/DouZero}"
DANZERO_DIR="${DANZERO_DIR:-$SCRIPT_DIR/Danzero_plus}"
DOUZERO_MODEL_DIR="${DOUZERO_MODEL_DIR:-$DOUZERO_DIR/baselines/douzero_WP}"

# ===========================================
# RLCard Teacher Model Paths (v5)
# ===========================================
# RLCard 内置 Model Zoo (不需要额外下载!)
# 参考: https://github.com/datamllab/rlcard#pre-trained-and-rule-based-models
#
# 可用模型:
# - leduc-holdem-cfr: CFR 预训练模型
# - leduc-holdem-rule-v1/v2: 规则模型
# - uno-rule-v1: UNO 规则模型
# - limit-holdem-rule-v1: Limit Hold'em 规则模型
# - gin-rummy-novice-rule: Gin Rummy 规则模型
# - doudizhu-rule-v1: 斗地主规则模型
#
# 这些模型通过 rlcard.models.load(model_name) 加载

# Teacher 模型配置 (使用 RLCard 内置模型)
declare -A RLCARD_TEACHER_MODELS=(
    ["uno"]="uno-rule-v1"
    ["gin"]="gin-rummy-novice-rule"
    ["leduc"]="leduc-holdem-cfr"          # 使用 CFR 预训练模型 (最强)
    ["limit"]="limit-holdem-rule-v1"       # 使用规则模型
    ["nolimit"]="random"                   # No-limit 没有内置强模型，用 random
)

# 备选: 如果你有自己训练的 DQN 模型
RLCARD_DQN_MODEL_BASE="${RLCARD_DQN_MODEL_BASE:-}"

# 检查是否有自定义 DQN 模型
_init_rlcard_teachers() {
    if [ -n "$RLCARD_DQN_MODEL_BASE" ] && [ -d "$RLCARD_DQN_MODEL_BASE" ]; then
        local dqn_path
        
        # 如果存在 DQN checkpoint，优先使用
        dqn_path="${RLCARD_DQN_MODEL_BASE}/leduc-holdem/model.pth"
        [ -f "$dqn_path" ] && RLCARD_TEACHER_MODELS["leduc"]="$dqn_path"
        
        dqn_path="${RLCARD_DQN_MODEL_BASE}/limit-holdem/model.pth"
        [ -f "$dqn_path" ] && RLCARD_TEACHER_MODELS["limit"]="$dqn_path"
        
        dqn_path="${RLCARD_DQN_MODEL_BASE}/no-limit-holdem/model.pth"
        [ -f "$dqn_path" ] && RLCARD_TEACHER_MODELS["nolimit"]="$dqn_path"
    fi
}

# 初始化
_init_rlcard_teachers

# RLCard 环境名映射
declare -A RLCARD_ENV_NAMES=(
    ["uno"]="uno"
    ["gin"]="gin-rummy"
    ["leduc"]="leduc-holdem"
    ["limit"]="limit-holdem"
    ["nolimit"]="no-limit-holdem"
)

# ===========================================
# Data generation parameters
# ===========================================
NUM_EPISODES="${NUM_EPISODES:-100000}"
NUM_WORKERS="${NUM_WORKERS:-5}"
DATA_GEN_SEED="${DATA_GEN_SEED:-43}"

# RLCard 默认数据量 (可通过环境变量覆盖)
RLCARD_UNO_EPISODES="${RLCARD_UNO_EPISODES:-50000}"
RLCARD_GIN_EPISODES="${RLCARD_GIN_EPISODES:-50000}"
RLCARD_LEDUC_EPISODES="${RLCARD_LEDUC_EPISODES:-400000}"
RLCARD_LIMIT_EPISODES="${RLCARD_LIMIT_EPISODES:-200000}"
RLCARD_NOLIMIT_EPISODES="${RLCARD_NOLIMIT_EPISODES:-400000}"

# Source environment for cloning
SOURCE_ENV="${SOURCE_ENV:-base}"

# Supported games (8 games from paper)
SUPPORTED_GAMES=("doudizhu" "guandan" "riichi" "uno" "gin" "leduc" "limit" "nolimit")

# RLCard games only
RLCARD_GAMES=("uno" "gin" "leduc" "limit" "nolimit")

# Game to convert_data.py mapping
declare -A GAME_CONVERT_MAP=(
    ["doudizhu"]="dou_dizhu"
    ["guandan"]="guandan"
    ["riichi"]="riichi"
    ["uno"]="uno"
    ["gin"]="gin_rummy"
    ["leduc"]="leduc_holdem"
    ["limit"]="limit_holdem"
    ["nolimit"]="nolimit_holdem"
)

# ===========================================
# Utility Functions
# ===========================================

print_header() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║     LLM4CardGame - Complete Pipeline (v5 - RLCard)        ║"
    echo "║     NeurIPS 2025: Can LLMs Master Complex Card Games?     ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""
}

print_step() {
    echo ""
    echo "┌──────────────────────────────────────────────────────────┐"
    echo "│  $1"
    echo "└──────────────────────────────────────────────────────────┘"
    echo ""
}

check_dir() {
    mkdir -p "$1"
}

log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1" >&2
}

log_warn() {
    echo "[WARN] $1" >&2
}

log_success() {
    echo "[✓] $1"
}

activate_env() {
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate ${CONDA_ENV_NAME} 2>/dev/null || {
            log_warn "Conda environment '${CONDA_ENV_NAME}' not found"
            return 0
        }
        log_info "Activated conda environment: ${CONDA_ENV_NAME}"
    else
        log_warn "Conda not found, using current Python environment"
    fi
}

check_server() {
    local port="${1:-$API_PORT}"
    curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1
}

wait_for_server() {
    local port="${1:-$API_PORT}"
    local max_wait="${2:-180}"
    local waited=0
    
    log_info "Waiting for server on port $port..."
    while [ $waited -lt $max_wait ]; do
        if check_server "$port"; then
            log_success "Server ready on port $port"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        echo -n "."
    done
    echo ""
    log_error "Server timeout (${max_wait}s)"
    return 1
}

# ===========================================
# HuggingFace Model Path Resolution
# ===========================================

get_hf_model_local_path() {
    local model_id="$1"
    local cache_model_id=$(echo "$model_id" | sed 's/\//-_-/g')
    
    local hub_path="$HF_CACHE_DIR/hub/models--${cache_model_id//-_-/--}"
    if [ -d "$hub_path/snapshots" ]; then
        local latest=$(ls -t "$hub_path/snapshots" 2>/dev/null | head -1)
        if [ -n "$latest" ] && [ -f "$hub_path/snapshots/$latest/config.json" ]; then
            echo "$hub_path/snapshots/$latest"
            return 0
        fi
    fi
    
    local models_path="$HF_CACHE_DIR/models--${cache_model_id//-_-/--}"
    if [ -d "$models_path/snapshots" ]; then
        local latest=$(ls -t "$models_path/snapshots" 2>/dev/null | head -1)
        if [ -n "$latest" ] && [ -f "$models_path/snapshots/$latest/config.json" ]; then
            echo "$models_path/snapshots/$latest"
            return 0
        fi
    fi
    
    return 1
}

validate_model_path() {
    local model_path="$1"
    
    if [ -z "$model_path" ]; then
        return 1
    fi
    
    if [[ "$model_path" == /* ]]; then
        if [ -f "$model_path/config.json" ]; then
            if python3 -c "import json; json.load(open('$model_path/config.json'))" 2>/dev/null; then
                return 0
            else
                log_error "Invalid config.json at $model_path"
                return 1
            fi
        fi
        return 1
    fi
    
    local local_path=$(get_hf_model_local_path "$model_path")
    if [ -n "$local_path" ] && [ -f "$local_path/config.json" ]; then
        return 0
    fi
    
    return 1
}

resolve_model_path() {
    local model_key_or_path="$1"
    
    if [[ -n "${MODEL_HF_PATHS[$model_key_or_path]}" ]]; then
        local model_id="${MODEL_HF_PATHS[$model_key_or_path]}"
        local local_path=$(get_hf_model_local_path "$model_id")
        if [ -n "$local_path" ]; then
            echo "$local_path"
            return 0
        fi
        echo "$model_id"
        return 0
    fi
    
    if [[ "$model_key_or_path" == /* ]] && [[ -d "$model_key_or_path" ]]; then
        if [ -f "$model_key_or_path/config.json" ]; then
            echo "$model_key_or_path"
            return 0
        fi
    fi
    
    local local_cache_path="$LOCAL_MODEL_CACHE/$model_key_or_path"
    if [[ -d "$local_cache_path" ]] && [[ -f "$local_cache_path/config.json" ]]; then
        echo "$local_cache_path"
        return 0
    fi
    
    local hf_local=$(get_hf_model_local_path "$model_key_or_path")
    if [ -n "$hf_local" ]; then
        echo "$hf_local"
        return 0
    fi
    
    echo "$model_key_or_path"
}

get_model_template() {
    local model_path="$1"
    
    for key in "${!MODEL_HF_PATHS[@]}"; do
        if [[ "$model_path" == *"${MODEL_HF_PATHS[$key]}"* ]] || [[ "$model_path" == *"$key"* ]]; then
            echo "${MODEL_TEMPLATES[$key]}"
            return 0
        fi
    done
    
    local lower_path=$(echo "$model_path" | tr '[:upper:]' '[:lower:]')
    if [[ "$lower_path" == *"qwen"* ]]; then
        echo "qwen"
    elif [[ "$lower_path" == *"glm"* ]] || [[ "$lower_path" == *"chatglm"* ]]; then
        echo "glm4"
    elif [[ "$lower_path" == *"llama"* ]]; then
        echo "llama3"
    else
        echo "default"
    fi
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Environment"
    
    if ! command -v conda &> /dev/null; then
        log_error "Conda not found. Install Miniconda/Anaconda first."
        exit 1
    fi
    
    if ! conda env list | grep -q "^${SOURCE_ENV} "; then
        log_error "Source environment '${SOURCE_ENV}' not found!"
        conda env list
        exit 1
    fi
    
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        log_info "Environment '${CONDA_ENV_NAME}' exists."
        read -p "Update packages? (Y/n): " choice
        if [[ "$choice" =~ ^[Nn]$ ]]; then
            log_info "Recreating environment..."
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            activate_env
            pip install -q rlcard douzero rlcard tenacity openai tqdm vllm llamafactory huggingface_hub 2>/dev/null || true
            log_success "Environment updated"
            return 0
        fi
    fi
    
    log_info "Cloning from '${SOURCE_ENV}' to '${CONDA_ENV_NAME}'..."
    conda create --name ${CONDA_ENV_NAME} --clone ${SOURCE_ENV} -y
    
    activate_env
    
    log_info "Installing packages..."
    pip install -q douzero rlcard tenacity openai tqdm vllm llamafactory huggingface_hub 2>/dev/null || true
    
    log_success "Environment setup complete: ${CONDA_ENV_NAME}"
}

# ===========================================
# Prepare Directories
# ===========================================

prepare_dirs() {
    print_step "Preparing Directories"
    
    check_dir "$OUTPUT_DIR"
    check_dir "$OUTPUT_DIR/logs"
    check_dir "$OUTPUT_DIR/eval_results"
    check_dir "$DATA_DIR/raw"
    check_dir "$DATA_DIR/sft"
    check_dir "$PROJECT_DIR/data_gen"
    check_dir "$DOUZERO_DIR"
    check_dir "$LOCAL_MODEL_CACHE"
    check_dir "$TRAIN_CONFIG_DIR"
    
    log_info "Project:     $PROJECT_DIR"
    log_info "Data:        $DATA_DIR"
    log_info "Output:      $OUTPUT_DIR"
    log_info "DouZero:     $DOUZERO_DIR"
    log_info "Model Cache: $LOCAL_MODEL_CACHE"
    log_info "HF Cache:    $HF_CACHE_DIR"
    log_success "Directories prepared"
}

# ===========================================
# Download Models
# ===========================================

download_base_model() {
    print_step "Downloading/Verifying Base Model"
    
    local model_id="$BASE_MODEL"
    local resolved_path=$(resolve_model_path "$model_id")
    
    if [[ "$resolved_path" == /* ]] && [ -f "$resolved_path/config.json" ]; then
        log_success "Model already cached: $resolved_path"
        return 0
    fi
    
    log_info "Downloading model from HuggingFace Hub: $model_id"
    log_info "Cache directory: $HF_CACHE_DIR"
    
    activate_env
    
    python3 << EOF
import os
import sys
from huggingface_hub import snapshot_download
import json

model_id = "$model_id"
cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

print(f"Downloading model: {model_id}")
print(f"Cache directory: {cache_dir}")

try:
    local_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_files_only=False,
        resume_download=True
    )
    print(f"Model downloaded to: {local_path}")
    
    config_path = os.path.join(local_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Model config valid: {config.get('model_type', 'unknown')}")
    else:
        print(f"⚠ Warning: config.json not found at {config_path}")
        sys.exit(1)
        
except Exception as e:
    print(f"Error downloading model: {e}")
    sys.exit(1)
EOF
    
    if [ $? -ne 0 ]; then
        log_error "Failed to download model: $model_id"
        exit 1
    fi
    
    log_success "Model ready: $model_id"
}

download_douzero_models() {
    print_step "Downloading DouZero Teacher Models"
    
    local models=("landlord.ckpt" "landlord_up.ckpt" "landlord_down.ckpt")
    local all_exist=true
    
    for model in "${models[@]}"; do
        if [ ! -f "$DOUZERO_MODEL_DIR/$model" ]; then
            all_exist=false
            break
        fi
    done
    
    if $all_exist; then
        log_success "DouZero models exist"
        return 0
    fi
    
    check_dir "$DOUZERO_MODEL_DIR"
    
    log_info "Downloading DouZero WP models..."
    local BASE_URL="https://github.com/kwai/DouZero/releases/download/v1.0"
    
    for model in "${models[@]}"; do
        local path="$DOUZERO_MODEL_DIR/$model"
        if [ ! -f "$path" ]; then
            log_info "  $model..."
            if command -v wget &> /dev/null; then
                wget -q --show-progress -O "$path" "$BASE_URL/$model" || true
            elif command -v curl &> /dev/null; then
                curl -L -o "$path" "$BASE_URL/$model" || true
            fi
        fi
    done
    
    if [ -f "$DOUZERO_MODEL_DIR/landlord.ckpt" ]; then
        log_success "DouZero models downloaded"
    else
        log_warn "Download failed. Manual: https://github.com/kwai/DouZero/releases"
    fi
}

# ===========================================
# Data Generation
# ===========================================

generate_training_data() {
    print_step "Generating Training Data"
    
    activate_env
    cd "$PROJECT_DIR"
    
    local game="${1:-doudizhu}"
    local num_episodes="${2:-$NUM_EPISODES}"
    local seed="${3:-$DATA_GEN_SEED}"
    
    log_info "Game: $game | Episodes: $num_episodes | Seed: $seed"
    log_info "GPU: $CUDA_DEVICE"
    
    case $game in
        doudizhu)
            _generate_doudizhu_data "$num_episodes" "$seed"
            ;;
        guandan)
            if [ -f "scripts/gen_data_guandan.sh" ]; then
                bash scripts/gen_data_guandan.sh
            else
                log_error "Guandan script missing"
                exit 1
            fi
            ;;
        riichi)
            log_warn "Riichi generation requires libriichi - skipping"
            ;;
        uno|gin|leduc|limit|nolimit)
            _generate_rlcard_data "$game" "$num_episodes" "$seed"
            ;;
        rlcard_all)
            log_info "Generating data for all 5 RLCard games..."
            for g in "${RLCARD_GAMES[@]}"; do
                local ep_var="RLCARD_${g^^}_EPISODES"
                local episodes="${!ep_var:-50000}"
                _generate_rlcard_data "$g" "$episodes" "$seed"
            done
            ;;
        *)
            log_error "Unknown game: $game"
            log_info "Supported: ${SUPPORTED_GAMES[*]}"
            exit 1
            ;;
    esac
    
    cd "$SCRIPT_DIR"
    log_success "Data generation complete: $game"
}

_generate_doudizhu_data() {
    local num_episodes="${1:-100000}"
    local seed="${2:-43}"
    local eval_data_name="eval_train_s${seed}"
    local eval_data_path="$DOUZERO_DIR/${eval_data_name}.pkl"
    local log_dir="$PROJECT_DIR/data_gen/douzero_train_s${seed}"
    
    log_info "=== Doudizhu Data Generation ==="
    log_info "Using DouZero teacher models"
    
    check_dir "$log_dir"
    
    if [ ! -f "$eval_data_path" ]; then
        log_info "Generating initial game states..."
        python util/douzero_util/generate_eval_data.py \
            --num_games "$num_episodes" \
            --output "$DOUZERO_DIR/${eval_data_name}" \
            --seed "$seed"
    fi
    
    log_info "Landlord: DouZero | Farmers: RLCard"
    python eval_llm_douzero.py \
        --eval_data "$eval_data_path" \
        --log_dir "$log_dir" \
        --landlord "$DOUZERO_MODEL_DIR/landlord.ckpt" \
        --landlord_up rlcard \
        --landlord_down rlcard \
        --gpu_device "$CUDA_DEVICE" \
        --num_workers "$NUM_WORKERS" \
        --seed "$seed"
    
    log_info "Landlord: RLCard | Farmers: DouZero"
    python eval_llm_douzero.py \
        --eval_data "$eval_data_path" \
        --log_dir "$log_dir" \
        --landlord rlcard \
        --landlord_up "$DOUZERO_MODEL_DIR/landlord_up.ckpt" \
        --landlord_down "$DOUZERO_MODEL_DIR/landlord_down.ckpt" \
        --gpu_device "$CUDA_DEVICE" \
        --num_workers "$NUM_WORKERS" \
        --seed "$((seed + 1))"
    
    local file_count=$(ls -1 "$log_dir"/trajectory-*.jsonl 2>/dev/null | wc -l)
    log_success "Generated $file_count trajectory files"
}

# ===========================================
# RLCard Data Generation (v5 - FIXED)
# ===========================================

_generate_rlcard_data() {
    local game="$1"
    local num_episodes="${2:-50000}"
    local seed="${3:-45}"
    
    # 获取环境名
    local env_name="${RLCARD_ENV_NAMES[$game]}"
    if [ -z "$env_name" ]; then
        log_error "Unknown RLCard game: $game"
        return 1
    fi
    
    # 获取 teacher 模型
    local teacher_model="${RLCARD_TEACHER_MODELS[$game]}"
    if [ -z "$teacher_model" ]; then
        log_error "No teacher model configured for: $game"
        return 1
    fi
    
    # 输出目录
    local output_dir="$PROJECT_DIR/data_gen/rlcard_${game}_s${seed}_${num_episodes}"
    check_dir "$output_dir"
    
    log_info "=== RLCard Data Generation: $game ==="
    log_info "Environment: $env_name"
    log_info "Teacher: $teacher_model"
    log_info "Episodes: $num_episodes"
    log_info "Output: $output_dir"
    
    # 检查 teacher 模型是否存在 (对于 DQN 模型)
    if [[ "$teacher_model" == *.pt ]]; then
        if [ ! -f "$teacher_model" ]; then
            log_warn "Teacher model not found: $teacher_model"
            log_info "Will try to use 'dqn' as model identifier..."
            teacher_model="dqn"
        fi
    fi
    
    # 运行数据生成
    python -m util.rlcard_util.gen_data \
        --env "$env_name" \
        --models "$teacher_model" random \
        --out_dir "$output_dir" \
        --num_games "$num_episodes" \
        --num_workers "$NUM_WORKERS" \
        --cuda "$CUDA_DEVICE" \
        --seed "$seed"
    
    local file_count=$(ls -1 "$output_dir"/*.txt 2>/dev/null | wc -l)
    log_success "Generated $file_count trajectory files for $game"
}

# ===========================================
# RLCard Teacher Model Check (v5)
# ===========================================

check_rlcard_teachers() {
    print_step "Checking RLCard Teacher Models"
    
    # 重新初始化
    _init_rlcard_teachers
    
    echo "RLCard Model Zoo (内置模型，无需下载):"
    echo "  https://github.com/datamllab/rlcard#pre-trained-and-rule-based-models"
    echo ""
    
    for _game in "${RLCARD_GAMES[@]}"; do
        local teacher="${RLCARD_TEACHER_MODELS[$_game]}"
        local status=""
        local note=""
        
        if [[ "$teacher" == *.pth ]] || [[ "$teacher" == *.pt ]]; then
            # 自定义 DQN 模型
            if [ -f "$teacher" ]; then
                status="✓ DQN (custom)"
            else
                status="✗ NOT FOUND"
            fi
        elif [[ "$teacher" == "random" ]]; then
            status="⚠ random"
            note="(no built-in strong model for this game)"
        elif [[ "$teacher" == *"-cfr" ]]; then
            status="✓ CFR (built-in, strongest)"
        elif [[ "$teacher" == *"-rule-"* ]]; then
            status="✓ Rule-based (built-in)"
        else
            status="✓ Built-in"
        fi
        
        printf "  %-10s: %-30s [%s]\n" "$_game" "$teacher" "$status"
        [ -n "$note" ] && printf "             %s\n" "$note"
    done
    
    echo ""
    log_success "RLCard 内置模型可直接使用，无需额外下载!"
    echo ""
    echo "如果你想使用自己训练的 DQN 模型:"
    echo "  export RLCARD_DQN_MODEL_BASE=/你的模型目录"
}

# ===========================================
# Data Conversion
# ===========================================

convert_training_data() {
    print_step "Converting to SFT Format"
    
    activate_env
    cd "$PROJECT_DIR"
    
    local game="${1:-all}"
    
    if [ "$game" = "all" ]; then
        for g in "${SUPPORTED_GAMES[@]}"; do
            _convert_single_game "$g"
        done
    elif [ "$game" = "rlcard_all" ]; then
        for g in "${RLCARD_GAMES[@]}"; do
            _convert_single_game "$g"
        done
    else
        _convert_single_game "$game"
    fi
    
    _merge_sft_files "$game"
    _validate_sft_data
    _create_dataset_info
    
    cd "$SCRIPT_DIR"
    log_success "Data conversion complete"
}

_convert_single_game() {
    local game="$1"
    local game_convert="${GAME_CONVERT_MAP[$game]:-$game}"
    
    # 确定输入目录
    local input_dir=""
    case $game in
        doudizhu) 
            input_dir="$PROJECT_DIR/data_gen/douzero_train_s${DATA_GEN_SEED}" 
            ;;
        guandan) 
            input_dir="$PROJECT_DIR/data_gen/guandan_train" 
            ;;
        riichi) 
            input_dir="$PROJECT_DIR/data_gen/riichi_train" 
            ;;
        uno|gin|leduc|limit|nolimit)
            # RLCard 游戏 - 查找最新的数据目录
            input_dir=$(find "$PROJECT_DIR/data_gen" -maxdepth 1 -type d -name "rlcard_${game}_s*" | sort -r | head -1)
            if [ -z "$input_dir" ]; then
                # 尝试旧格式
                input_dir="$PROJECT_DIR/data_gen/${game}_train_s${DATA_GEN_SEED}"
            fi
            ;;
    esac
    
    if [ ! -d "$input_dir" ]; then
        log_warn "No data for $game at $input_dir"
        return 0
    fi
    
    local output_file="$DATA_DIR/sft/sft-${game}.jsonl"
    
    log_info "Converting: $game"
    log_info "  Input:  $input_dir"
    log_info "  Output: $output_file"
    log_info "  Game type: $game_convert"
    
    python convert_data.py \
        --game "$game_convert" \
        --input "$input_dir" \
        --output "$output_file"
    
    if [ -f "$output_file" ]; then
        local count=$(wc -l < "$output_file")
        log_success "  $count samples converted"
    else
        log_warn "  No output generated"
    fi
}

_merge_sft_files() {
    local game="${1:-all}"
    
    log_info "Merging SFT files..."
    
    local train_file="$DATA_DIR/sft/train.jsonl"
    
    # 如果是单个游戏，创建游戏特定的训练文件
    if [ "$game" != "all" ] && [ "$game" != "rlcard_all" ]; then
        train_file="$DATA_DIR/sft/train_${game}.jsonl"
    fi
    
    rm -f "$train_file"
    
    local total=0
    local pattern="sft-*.jsonl"
    
    # 根据游戏类型选择要合并的文件
    if [ "$game" = "rlcard_all" ]; then
        for g in "${RLCARD_GAMES[@]}"; do
            local f="$DATA_DIR/sft/sft-${g}.jsonl"
            if [ -f "$f" ]; then
                cat "$f" >> "$train_file"
                local count=$(wc -l < "$f")
                total=$((total + count))
                log_info "  Added $g: $count samples"
            fi
        done
    elif [ "$game" != "all" ]; then
        local f="$DATA_DIR/sft/sft-${game}.jsonl"
        if [ -f "$f" ]; then
            cat "$f" >> "$train_file"
            total=$(wc -l < "$f")
        fi
    else
        for f in "$DATA_DIR/sft"/sft-*.jsonl; do
            if [ -f "$f" ]; then
                cat "$f" >> "$train_file"
                local count=$(wc -l < "$f")
                total=$((total + count))
            fi
        done
    fi
    
    log_success "Merged $total samples -> $train_file"
}

_validate_sft_data() {
    log_info "Validating SFT data..."
    
    export DATA_DIR="$DATA_DIR"
    python3 << 'VALIDATE_SCRIPT'
import json, os, sys

data_dir = os.environ.get('DATA_DIR', './data')

# 检查所有 train*.jsonl 文件
for train_file in sorted(os.listdir(f"{data_dir}/sft")):
    if not train_file.startswith('train') or not train_file.endswith('.jsonl'):
        continue
    
    filepath = f"{data_dir}/sft/{train_file}"
    total, valid, empty_legal, games = 0, 0, 0, {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            try:
                item = json.loads(line.strip())
                inst = item.get('instruction', '')
                
                for key, name in [
                    ('Dou Dizhu', 'doudizhu'), ('Guandan', 'guandan'),
                    ('Mahjong', 'riichi'), ('UNO', 'uno'),
                    ('Gin Rummy', 'gin'), ('Leduc', 'leduc'),
                    ('Limit Hold', 'limit'), ('No-Limit', 'nolimit')
                ]:
                    if key in inst:
                        games[name] = games.get(name, 0) + 1
                        break
                
                if 'legal_actions": []' in inst:
                    empty_legal += 1
                else:
                    valid += 1
            except:
                pass

    if total > 0:
        print(f"\n{'='*50}")
        print(f"File: {train_file}")
        print(f"{'='*50}")
        print(f"Total:       {total:,}")
        print(f"Valid:       {valid:,} ({valid/total*100:.1f}%)")
        print(f"Empty legal: {empty_legal:,} ({empty_legal/total*100:.1f}%)")
        if games:
            print("\nPer game:")
            for g, c in sorted(games.items(), key=lambda x: -x[1]):
                print(f"  {g:12s}: {c:,}")
        print("✓ OK" if empty_legal < total * 0.05 else "⚠ High invalid rate!")
VALIDATE_SCRIPT
}

_create_dataset_info() {
    log_info "Creating dataset_info.json..."
    
    # 创建包含所有数据集的 dataset_info.json
    python3 << EOF
import json
import os

data_dir = "$DATA_DIR"
sft_dir = os.path.join(data_dir, "sft")

dataset_info = {}

# 主训练集
if os.path.exists(os.path.join(sft_dir, "train.jsonl")):
    dataset_info["card_sft"] = {
        "file_name": "sft/train.jsonl",
        "columns": {
            "prompt": "instruction",
            "response": "output"
        }
    }

# 按游戏分类的训练集
for f in os.listdir(sft_dir):
    if f.startswith("train_") and f.endswith(".jsonl"):
        game = f.replace("train_", "").replace(".jsonl", "")
        dataset_info[f"card_sft_{game}"] = {
            "file_name": f"sft/{f}",
            "columns": {
                "prompt": "instruction",
                "response": "output"
            }
        }

# 按数据量分类的训练集
for f in os.listdir(sft_dir):
    if f.startswith("train_") and f.endswith(".jsonl"):
        parts = f.replace("train_", "").replace(".jsonl", "")
        if parts.isdigit():
            dataset_info[f"card_sft_{parts}"] = {
                "file_name": f"sft/{f}",
                "columns": {
                    "prompt": "instruction",
                    "response": "output"
                }
            }

with open(os.path.join(data_dir, "dataset_info.json"), "w") as f:
    json.dump(dataset_info, f, indent=2, ensure_ascii=False)

print(f"Created dataset_info.json with {len(dataset_info)} datasets")
for name in sorted(dataset_info.keys()):
    print(f"  - {name}")
EOF
    
    log_success "Created dataset_info.json"
}

# ===========================================
# Data Splitting
# ===========================================

DATA_SIZE="${DATA_SIZE:-1000000}"

split_training_data() {
    local target_size="${1:-$DATA_SIZE}"
    local source_file="${2:-$DATA_DIR/sft/train.jsonl}"
    local output_file="$DATA_DIR/sft/train_${target_size}.jsonl"
    
    log_info "=== Data Splitting ===" >&2
    
    if [ ! -f "$source_file" ]; then
        log_error "Source data not found: $source_file"
        return 1
    fi
    
    if [ -f "$output_file" ]; then
        local existing=$(wc -l < "$output_file")
        if [ "$existing" -ge "$target_size" ]; then
            log_info "Split data exists: $output_file ($existing samples)" >&2
            echo "$output_file"
            return 0
        fi
    fi
    
    local total=$(wc -l < "$source_file")
    log_info "Source: $total samples" >&2
    log_info "Target: $target_size samples" >&2
    
    if [ "$total" -le "$target_size" ]; then
        log_warn "Source <= target, using all data"
        cp "$source_file" "$output_file"
    else
        log_info "Random sampling $target_size samples..." >&2
        shuf -n "$target_size" "$source_file" > "$output_file"
    fi
    
    local output_count=$(wc -l < "$output_file")
    log_success "Split complete: $output_file ($output_count samples)"
    
    update_dataset_info "$target_size"
    
    echo "$output_file"
}

update_dataset_info() {
    local target_size="$1"
    local dataset_info="$DATA_DIR/dataset_info.json"
    local dataset_name="card_sft_${target_size}"
    
    if grep -q "\"$dataset_name\"" "$dataset_info" 2>/dev/null; then
        log_info "dataset_info.json already contains $dataset_name"
        return 0
    fi
    
    log_info "Updating dataset_info.json..."
    
    python3 << EOF
import json

path = "$dataset_info"
name = "$dataset_name"
size = "$target_size"

with open(path, 'r') as f:
    data = json.load(f)

data[name] = {
    "file_name": f"sft/train_{size}.jsonl",
    "columns": {
        "prompt": "instruction",
        "response": "output"
    }
}

with open(path, 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Added {name}")
EOF
    
    log_success "dataset_info.json updated"
}

# ===========================================
# Training Configuration
# ===========================================

prepare_train_config() {
    local model_path="$1"
    local template="$2"
    local output_lora_dir="$3"
    local data_size="${4:-$DATA_SIZE}"
    local game="${5:-all}"
    
    local base_config="$TRAIN_CONFIG_DIR/card_lora_sft.yaml"
    local temp_config="$TRAIN_CONFIG_DIR/temp_${template}_${game}_${data_size}_train.yaml"
    
    # 确定数据集名称
    local dataset_name="card_sft"
    if [ "$game" != "all" ]; then
        dataset_name="card_sft_${game}"
    fi
    if [ "$data_size" != "all" ]; then
        dataset_name="card_sft_${data_size}"
    fi
    
    if [ ! -f "$base_config" ]; then
        echo "[ERROR] Base config not found: $base_config" >&2
        echo "[INFO] Creating default config..." >&2
        
        cat > "$base_config" << 'DEFAULTCFG'
### model
model_name_or_path: 'Qwen/Qwen2.5-7B-Instruct'
dataset_dir: '/data/jiacheng/system/cache/temp/icml2026/walking/LLM4CardGame/data'

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target: all

### dataset
dataset: card_sft
template: qwen
cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: '/data/jiacheng/system/cache/temp/icml2026/walking/LLM4CardGame/output/sft_lora'
logging_steps: 10
save_strategy: steps
save_only_model: true
save_steps: 500
save_total_limit: 5
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 16
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 1
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
gradient_checkpointing: true
optim: adamw_torch_fused

### eval
val_size: 0.01
per_device_eval_batch_size: 8
eval_strategy: steps
eval_steps: 500
DEFAULTCFG
    fi
    
    echo "[INFO] Preparing config from: $base_config" >&2
    
    cp "$base_config" "$temp_config"
    
    local escaped_model_path=$(echo "$model_path" | sed 's/[\/&]/\\&/g')
    local escaped_data_dir=$(echo "$DATA_DIR" | sed 's/[\/&]/\\&/g')
    local escaped_output_dir=$(echo "$output_lora_dir" | sed 's/[\/&]/\\&/g')
    
    sed -i "s|^model_name_or_path:.*|model_name_or_path: '${escaped_model_path}'|" "$temp_config"
    sed -i "s|^template:.*|template: ${template}|" "$temp_config"
    sed -i "s|^dataset:.*|dataset: ${dataset_name}|" "$temp_config"
    sed -i "s|^dataset_dir:.*|dataset_dir: '${escaped_data_dir}'|" "$temp_config"
    sed -i "s|^output_dir:.*|output_dir: '${escaped_output_dir}'|" "$temp_config"
    sed -i '/^report_to:/d' "$temp_config"
    
    if [ -s "$temp_config" ]; then
        if [ "$(tail -c 1 "$temp_config" | wc -l)" -eq 0 ]; then
            echo "" >> "$temp_config"
        fi
    fi
    
    echo "[INFO] Config prepared: $temp_config" >&2
    echo "[INFO]   model: $model_path" >&2
    echo "[INFO]   template: $template" >&2
    echo "[INFO]   dataset: $dataset_name" >&2
    echo "[INFO]   output: $output_lora_dir" >&2
    
    echo "$temp_config"
}

# ===========================================
# Training
# ===========================================

run_training() {
    print_step "Training Model"
    
    activate_env
    cd "$PROJECT_DIR"
    
    local config="${1:-}"
    local data_size="${2:-$DATA_SIZE}"
    local game="${3:-all}"
    
    local model_path=$(resolve_model_path "$BASE_MODEL")
    local template=$(get_model_template "$model_path")
    local output_lora_dir="$OUTPUT_DIR/sft_lora_${template}_${game}_${data_size}"
    
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
    check_dir "$OUTPUT_DIR/logs"
    
    log_info "Model key: $MODEL_KEY"
    log_info "Base model: $BASE_MODEL"
    log_info "Resolved path: $model_path"
    log_info "Template: $template"
    log_info "Game: $game"
    log_info "Data size: $data_size"
    log_info "Output: $output_lora_dir"
    
    if ! validate_model_path "$model_path"; then
        log_warn "Model not properly cached, downloading..."
        download_base_model
        model_path=$(resolve_model_path "$BASE_MODEL")
        
        if ! validate_model_path "$model_path"; then
            log_error "Model validation failed after download"
            exit 1
        fi
    fi
    log_success "Model validated: $model_path"
    
    # 数据切割
    local data_file=""
    if [ "$data_size" != "all" ]; then
        local source_file="$DATA_DIR/sft/train.jsonl"
        if [ "$game" != "all" ]; then
            source_file="$DATA_DIR/sft/train_${game}.jsonl"
        fi
        
        log_info "Splitting training data to $data_size samples..."
        data_file=$(split_training_data "$data_size" "$source_file")
        if [ -z "$data_file" ] || [ ! -f "$data_file" ]; then
            log_error "Failed to split training data"
            exit 1
        fi
        log_success "Using data file: $data_file"
    else
        if [ "$game" != "all" ]; then
            data_file="$DATA_DIR/sft/train_${game}.jsonl"
        else
            data_file="$DATA_DIR/sft/train.jsonl"
        fi
    fi
    
    if [ ! -f "$data_file" ]; then
        log_error "Training data missing: $data_file"
        exit 1
    fi
    
    local sample_count=$(wc -l < "$data_file")
    log_info "Training samples: $sample_count"
    
    if [ -z "$config" ] || [ ! -f "$config" ]; then
        config=$(prepare_train_config "$model_path" "$template" "$output_lora_dir" "$data_size" "$game")
    fi
    
    if [ -z "$config" ] || [ ! -f "$config" ]; then
        log_error "Failed to prepare training config"
        exit 1
    fi
    
    log_info "Using config: $config"
    
    echo "--- Config Contents ---"
    cat "$config"
    echo ""
    echo "--- End Config ---"
    
    log_info "Validating YAML syntax..."
    if ! python3 -c "import yaml; yaml.safe_load(open('$config'))" 2>/dev/null; then
        log_error "Invalid YAML syntax in config"
        python3 -c "import yaml; yaml.safe_load(open('$config'))" 2>&1 || true
        exit 1
    fi
    log_success "YAML syntax valid"
    
    if ! command -v llamafactory-cli &> /dev/null; then
        log_error "LLaMA-Factory not installed"
        exit 1
    fi
    
    local batch_size=$(grep "^per_device_train_batch_size" "$config" | head -1 | awk '{print $2}')
    local grad_accum=$(grep "^gradient_accumulation_steps" "$config" | head -1 | awk '{print $2}')
    local effective_batch=$((batch_size * grad_accum))
    local expected_steps=$((sample_count / effective_batch))
    log_info "Effective batch size: $effective_batch"
    log_info "Expected steps: $expected_steps"
    log_info "Estimated time: ~$((expected_steps * 5 / 3600)) hours"
    
    local log_file="$OUTPUT_DIR/logs/train_${game}_$(date +%Y%m%d_%H%M%S).log"
    log_info "Starting training..."
    log_info "Log file: $log_file"
    
    llamafactory-cli train "$config" 2>&1 | tee "$log_file"
    local train_status=${PIPESTATUS[0]}
    
    if [[ "$config" == *"temp_"* ]]; then
        rm -f "$config"
        log_info "Cleaned up temp config"
    fi
    
    cd "$SCRIPT_DIR"
    
    if [ $train_status -eq 0 ]; then
        log_success "Training complete!"
        log_info "Model saved to: $output_lora_dir"
    else
        log_error "Training failed with status $train_status"
        exit $train_status
    fi
}

# ===========================================
# Model Serving
# ===========================================

serve_model() {
    print_step "Starting Model Server"
    
    activate_env
    
    local model="${1:-$(resolve_model_path $BASE_MODEL)}"
    local template=$(get_model_template "$model")
    local lora="${2:-$OUTPUT_DIR/sft_lora_${template}}"
    local port="${3:-$API_PORT}"
    local mode="${4:-lora}"
    
    if check_server "$port"; then
        log_info "Server already running on $port"
        return 0
    fi
    
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
    
    log_info "Model: $model"
    log_info "LoRA:  $lora"
    log_info "Port:  $port"
    log_info "Mode:  $mode"
    log_info "GPU:   $CUDA_DEVICE"
    
    if ! python -c "import vllm" 2>/dev/null; then
        log_error "vLLM not installed"
        exit 1
    fi
    
    if [ "$mode" = "lora" ] && [ -d "$lora" ]; then
        python -m vllm.entrypoints.openai.api_server \
            --model "$model" \
            --enable-lora --max-lora-rank 64 \
            --lora-modules "llm-lora=$lora" \
            --port "$port" \
            --trust-remote-code \
            --max-model-len 16384 \
            --gpu-memory-utilization 0.92 &
    else
        log_warn "LoRA path not found: $lora, serving base model"
        python -m vllm.entrypoints.openai.api_server \
            --model "$model" \
            --port "$port" \
            --trust-remote-code \
            --max-model-len 16384 \
            --gpu-memory-utilization 0.92 &
    fi
    
    local pid=$!
    echo $pid > /tmp/llm4cardgame_server.pid
    log_info "Server PID: $pid"
    
    wait_for_server "$port" 180
}

stop_server() {
    print_step "Stopping Server"
    
    if [ -f /tmp/llm4cardgame_server.pid ]; then
        local pid=$(cat /tmp/llm4cardgame_server.pid)
        kill $pid 2>/dev/null && log_info "Killed PID $pid"
        rm -f /tmp/llm4cardgame_server.pid
    fi
    
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    log_success "Server stopped"
}

# ===========================================
# Evaluation
# ===========================================

generate_eval_data() {
    print_step "Generating Eval Data"
    
    activate_env
    cd "$PROJECT_DIR"
    
    local game="${1:-doudizhu}"
    local num_games="${2:-100}"
    local seed="${3:-44}"
    
    case $game in
        doudizhu)
            local output="$DOUZERO_DIR/eval_s${seed}-${num_games}"
            if [ -f "${output}.pkl" ]; then
                log_info "Eval data exists"
                return 0
            fi
            
            python util/douzero_util/generate_eval_data.py \
                --num_games "$num_games" \
                --output "$output" \
                --seed "$seed"
            ;;
        *)
            log_info "RLCard games use on-the-fly evaluation"
            ;;
    esac
    
    cd "$SCRIPT_DIR"
    log_success "Eval data generated"
}

run_eval() {
    print_step "Running Evaluation"
    
    activate_env
    cd "$PROJECT_DIR"
    
    local game="${1:-doudizhu}"
    local seed="${2:-44}"
    local num_games="${3:-100}"
    local player1="${4:-llm}"
    local player2="${5:-random}"
    local log_dir="${6:-$OUTPUT_DIR/eval_results/${game}_${player1}_vs_${player2}_$(date +%Y%m%d_%H%M%S)}"
    
    check_dir "$log_dir"
    
    log_info "Game: $game"
    log_info "Players: $player1 vs $player2"
    log_info "Games: $num_games"
    log_info "Log dir: $log_dir"
    
    case $game in
        doudizhu)
            _eval_doudizhu "$seed" "$num_games" "$player1" "$player2" "$log_dir"
            ;;
        guandan)
            _eval_guandan "$num_games" "$player1" "$player2" "$log_dir"
            ;;
        riichi)
            log_warn "Riichi evaluation requires libriichi - skipping"
            ;;
        uno|gin|leduc|limit|nolimit)
            _eval_rlcard "$game" "$num_games" "$player1" "$player2" "$log_dir"
            ;;
        *)
            log_error "Unknown game: $game"
            exit 1
            ;;
    esac
    
    cd "$SCRIPT_DIR"
    log_success "Evaluation complete"
}

_eval_doudizhu() {
    local seed="$1"
    local num_games="$2"
    local player1="$3"
    local player2="$4"
    local log_dir="$5"
    
    local eval_data="$DOUZERO_DIR/eval_s${seed}-${num_games}.pkl"
    
    if [ ! -f "$eval_data" ]; then
        log_info "Generating eval data..."
        python util/douzero_util/generate_eval_data.py \
            --num_games "$num_games" \
            --output "$DOUZERO_DIR/eval_s${seed}-${num_games}" \
            --seed "$seed"
    fi
    
    local landlord_arg=""
    local landlord_up_arg=""
    local landlord_down_arg=""
    
    case $player1 in
        llm)
            landlord_arg="llm"
            ;;
        pre|teacher)
            landlord_arg="$DOUZERO_MODEL_DIR/landlord.ckpt"
            ;;
        rlcard|rule)
            landlord_arg="rlcard"
            ;;
        random)
            landlord_arg="random"
            ;;
    esac
    
    case $player2 in
        llm)
            landlord_up_arg="llm"
            landlord_down_arg="llm"
            ;;
        pre|teacher)
            landlord_up_arg="$DOUZERO_MODEL_DIR/landlord_up.ckpt"
            landlord_down_arg="$DOUZERO_MODEL_DIR/landlord_down.ckpt"
            ;;
        rlcard|rule)
            landlord_up_arg="rlcard"
            landlord_down_arg="rlcard"
            ;;
        random)
            landlord_up_arg="random"
            landlord_down_arg="random"
            ;;
    esac
    
    python eval_llm_douzero.py \
        --eval_data "$eval_data" \
        --log_dir "$log_dir" \
        --landlord "$landlord_arg" \
        --landlord_up "$landlord_up_arg" \
        --landlord_down "$landlord_down_arg" \
        --gpu_device "$CUDA_DEVICE" \
        --num_workers 1 \
        --seed "$seed"
}

_eval_guandan() {
    local num_games="$1"
    local player1="$2"
    local player2="$3"
    local log_dir="$4"
    
    if [ -f "scripts/eval_guandan.sh" ]; then
        bash scripts/eval_guandan.sh 44 "$num_games" "$CUDA_DEVICE" "$player1" "$player2" "$log_dir" "$log_dir/eval"
    else
        log_error "Guandan eval script missing"
    fi
}

# ===========================================
# RLCard Evaluation (v5 - FIXED)
# ===========================================

_eval_rlcard() {
    local game="$1"
    local num_games="$2"
    local player1="$3"
    local player2="$4"
    local log_dir="$5"
    
    local env_name="${RLCARD_ENV_NAMES[$game]}"
    if [ -z "$env_name" ]; then
        log_error "Unknown RLCard game: $game"
        return 1
    fi
    
    log_info "=== RLCard Evaluation: $game ==="
    log_info "Environment: $env_name"
    log_info "Player 1: $player1"
    log_info "Player 2: $player2"
    log_info "Games: $num_games"
    
    # 转换 player 参数为 gen_data.py 格式
    local model1="$player1"
    local model2="$player2"
    
    # 处理 player1
    case $player1 in
        llm)
            model1="llm-lora"  # 使用 vLLM server
            ;;
        pre|teacher|dqn)
            model1="${RLCARD_TEACHER_MODELS[$game]}"
            ;;
        rule)
            case $game in
                uno) model1="uno-rule-v1" ;;
                gin) model1="gin-rummy-novice-rule" ;;
                *) model1="random" ;;
            esac
            ;;
    esac
    
    # 处理 player2
    case $player2 in
        llm)
            model2="llm-lora"
            ;;
        pre|teacher|dqn)
            model2="${RLCARD_TEACHER_MODELS[$game]}"
            ;;
        rule)
            case $game in
                uno) model2="uno-rule-v1" ;;
                gin) model2="gin-rummy-novice-rule" ;;
                *) model2="random" ;;
            esac
            ;;
    esac
    
    log_info "Model 1: $model1"
    log_info "Model 2: $model2"
    
    python -m util.rlcard_util.gen_data \
        --env "$env_name" \
        --models "$model1" "$model2" \
        --out_dir "$log_dir" \
        --num_games "$num_games" \
        --num_workers 1 \
        --cuda "$CUDA_DEVICE" \
        --seed 44
}

# Quick eval shortcuts
eval_vs_random() {
    run_eval "${1:-doudizhu}" 44 "${2:-100}" "llm" "random"
}

eval_vs_rule() {
    run_eval "${1:-doudizhu}" 44 "${2:-100}" "llm" "rule"
}

eval_vs_teacher() {
    run_eval "${1:-doudizhu}" 44 "${2:-100}" "llm" "teacher"
}

# ===========================================
# Full Pipelines
# ===========================================

run_full_train() {
    print_step "Full Training Pipeline: $1"
    
    local game="${1:-doudizhu}"
    local num_episodes="${2:-$NUM_EPISODES}"
    local data_size="${3:-$DATA_SIZE}"
    
    log_info "Game: $game | Episodes: $num_episodes | Data size: $data_size"
    log_info "Model: $BASE_MODEL (key: $MODEL_KEY)"
    
    prepare_dirs
    
    if [ "$game" = "doudizhu" ]; then
        download_douzero_models
    fi
    download_base_model
    
    generate_training_data "$game" "$num_episodes"
    convert_training_data "$game"
    run_training "" "$data_size" "$game"
    
    log_success "Full training pipeline complete"
}

run_full_eval() {
    print_step "Full Evaluation Pipeline"
    
    local game="${1:-doudizhu}"
    local num_games="${2:-100}"
    local opponent="${3:-random}"
    local template=$(get_model_template "$(resolve_model_path $BASE_MODEL)")
    local data_size="${4:-$DATA_SIZE}"
    local lora_path="$OUTPUT_DIR/sft_lora_${template}_${game}_${data_size}"
    
    # 如果指定的 lora 不存在，尝试其他路径
    if [ ! -d "$lora_path" ]; then
        lora_path="$OUTPUT_DIR/sft_lora_${template}"
    fi
    
    log_info "Game: $game"
    log_info "Opponent: $opponent"
    log_info "LoRA: $lora_path"
    
    prepare_dirs
    generate_eval_data "$game" "$num_games" 44
    serve_model "$(resolve_model_path $BASE_MODEL)" "$lora_path" "$API_PORT" "lora"
    run_eval "$game" 44 "$num_games" "llm" "$opponent"
    stop_server
    
    log_success "Evaluation pipeline complete"
}

run_full_pipeline() {
    print_step "Complete Pipeline"
    
    local game="${1:-doudizhu}"
    local train_ep="${2:-$NUM_EPISODES}"
    local eval_games="${3:-100}"
    local data_size="${4:-$DATA_SIZE}"
    
    run_full_train "$game" "$train_ep" "$data_size"
    
    generate_eval_data "$game" "$eval_games" 44
    local template=$(get_model_template "$(resolve_model_path $BASE_MODEL)")
    local lora_path="$OUTPUT_DIR/sft_lora_${template}_${game}_${data_size}"
    serve_model "$(resolve_model_path $BASE_MODEL)" "$lora_path" "$API_PORT" "lora"
    eval_vs_random "$game" "$eval_games"
    stop_server
    
    log_success "Complete pipeline finished"
}

# ===========================================
# RLCard Full Pipeline (v5)
# ===========================================

rlcard_full() {
    print_step "RLCard Full Pipeline"
    
    local game="${1:-uno}"
    local data_size="${2:-50000}"
    local eval_games="${3:-100}"
    
    # 验证是 RLCard 游戏
    if [[ ! " ${RLCARD_GAMES[*]} " =~ " ${game} " ]]; then
        log_error "Not an RLCard game: $game"
        log_info "Available: ${RLCARD_GAMES[*]}"
        exit 1
    fi
    
    log_info "=== RLCard Full Pipeline: $game ==="
    log_info "Data size: $data_size"
    log_info "Eval games: $eval_games"
    log_info "Model: $MODEL_KEY"
    
    # 确定数据量
    local ep_var="RLCARD_${game^^}_EPISODES"
    local num_episodes="${!ep_var:-50000}"
    
    # 1. 准备
    prepare_dirs
    download_base_model
    check_rlcard_teachers
    
    # 2. 数据生成
    generate_training_data "$game" "$num_episodes"
    
    # 3. 数据转换
    convert_training_data "$game"
    
    # 4. 训练
    run_training "" "$data_size" "$game"
    
    # 5. 评估
    local template=$(get_model_template "$(resolve_model_path $BASE_MODEL)")
    local lora_path="$OUTPUT_DIR/sft_lora_${template}_${game}_${data_size}"
    
    log_info "Starting evaluation..."
    serve_model "$(resolve_model_path $BASE_MODEL)" "$lora_path" "$API_PORT" "lora"
    
    # LLM vs Random
    run_eval "$game" 44 "$eval_games" "llm" "random"
    
    # LLM vs Rule (for uno/gin)
    if [ "$game" = "uno" ] || [ "$game" = "gin" ]; then
        run_eval "$game" 44 "$eval_games" "llm" "rule"
    fi
    
    # LLM vs Teacher (for poker games)
    if [ "$game" = "leduc" ] || [ "$game" = "limit" ] || [ "$game" = "nolimit" ]; then
        run_eval "$game" 44 "$eval_games" "llm" "teacher"
    fi
    
    stop_server
    
    log_success "RLCard full pipeline complete: $game"
}

rlcard_all_games() {
    print_step "RLCard All Games Pipeline"
    
    local data_size="${1:-50000}"
    local eval_games="${2:-100}"
    
    for game in "${RLCARD_GAMES[@]}"; do
        log_info "Processing game: $game"
        rlcard_full "$game" "$data_size" "$eval_games"
    done
    
    log_success "All RLCard games complete"
}

# ===========================================
# Status and Debug
# ===========================================

show_data_status() {
    print_step "Data Status"
    
    echo "=== Raw Data (data_gen/) ==="
    for game in "${SUPPORTED_GAMES[@]}"; do
        local count=0
        local dir=""
        
        case $game in
            doudizhu)
                dir="$PROJECT_DIR/data_gen/douzero_train_s${DATA_GEN_SEED}"
                if [ -d "$dir" ]; then
                    count=$(ls -1 "$dir"/*.jsonl 2>/dev/null | wc -l)
                fi
                ;;
            uno|gin|leduc|limit|nolimit)
                dir=$(find "$PROJECT_DIR/data_gen" -maxdepth 1 -type d -name "rlcard_${game}_s*" 2>/dev/null | sort -r | head -1)
                if [ -d "$dir" ]; then
                    count=$(ls -1 "$dir"/*.txt 2>/dev/null | wc -l)
                fi
                ;;
        esac
        
        if [ $count -gt 0 ]; then
            printf "  %-10s: %d files in %s\n" "$game" "$count" "$dir"
        else
            printf "  %-10s: (no data)\n" "$game"
        fi
    done
    
    echo ""
    echo "=== SFT Data (data/sft/) ==="
    if [ -d "$DATA_DIR/sft" ]; then
        for f in "$DATA_DIR/sft"/*.jsonl; do
            if [ -f "$f" ]; then
                local count=$(wc -l < "$f")
                local name=$(basename "$f")
                printf "  %-25s: %'d samples\n" "$name" "$count"
            fi
        done
    else
        echo "  (no SFT data)"
    fi
    
    echo ""
    echo "=== Trained Models (output/) ==="
    if [ -d "$OUTPUT_DIR" ]; then
        for d in "$OUTPUT_DIR"/sft_lora_*; do
            if [ -d "$d" ]; then
                local name=$(basename "$d")
                local ckpts=$(ls -1 "$d"/checkpoint-* 2>/dev/null | wc -l)
                printf "  %-40s: %d checkpoints\n" "$name" "$ckpts"
            fi
        done
    else
        echo "  (no trained models)"
    fi
}

test_server() {
    print_step "Testing Server"
    
    local port="${1:-$API_PORT}"
    
    if ! check_server "$port"; then
        log_error "Server not running on $port"
        exit 1
    fi
    
    log_success "Server running on $port"
    
    log_info "Models:"
    curl -s "http://localhost:$port/v1/models" | python -m json.tool 2>/dev/null || echo "Failed"
    
    log_info "Test completion:"
    curl -s "http://localhost:$port/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "llm-lora",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 20
        }' | python -m json.tool 2>/dev/null || echo "Failed"
}

show_config() {
    print_step "Configuration"
    
    local model_path=$(resolve_model_path "$BASE_MODEL")
    local template=$(get_model_template "$model_path")
    
    echo "=== Paths ==="
    echo "SCRIPT_DIR:       $SCRIPT_DIR"
    echo "PROJECT_DIR:      $PROJECT_DIR"
    echo "DATA_DIR:         $DATA_DIR"
    echo "OUTPUT_DIR:       $OUTPUT_DIR"
    echo "TRAIN_CONFIG_DIR: $TRAIN_CONFIG_DIR"
    echo "HF_CACHE_DIR:     $HF_CACHE_DIR"
    echo ""
    echo "=== Model ==="
    echo "MODEL_KEY:   $MODEL_KEY"
    echo "BASE_MODEL:  $BASE_MODEL"
    echo "Resolved:    $model_path"
    echo "Template:    $template"
    echo ""
    echo "=== Server ==="
    echo "CUDA_DEVICE: $CUDA_DEVICE"
    echo "API_PORT:    $API_PORT"
    echo "CONDA_ENV:   $CONDA_ENV_NAME"
    echo ""
    echo "=== Games ==="
    echo "All:     ${SUPPORTED_GAMES[*]}"
    echo "RLCard:  ${RLCARD_GAMES[*]}"
    echo ""
    echo "=== RLCard Teacher Models ==="
    for game in "${RLCARD_GAMES[@]}"; do
        local teacher="${RLCARD_TEACHER_MODELS[$game]}"
        local status="✗"
        if [[ "$teacher" == *.pt ]]; then
            [ -f "$teacher" ] && status="✓"
        else
            status="✓ (rule)"
        fi
        printf "  %-10s: %s [%s]\n" "$game" "$teacher" "$status"
    done
    echo ""
    echo "=== Data Generation Defaults ==="
    echo "UNO:     $RLCARD_UNO_EPISODES episodes"
    echo "GIN:     $RLCARD_GIN_EPISODES episodes"
    echo "LEDUC:   $RLCARD_LEDUC_EPISODES episodes"
    echo "LIMIT:   $RLCARD_LIMIT_EPISODES episodes"
    echo "NOLIMIT: $RLCARD_NOLIMIT_EPISODES episodes"
}

quick_fix() {
    activate_env
    pip install -q vllm openai tenacity rlcard douzero tqdm llamafactory huggingface_hub
    prepare_dirs
    log_success "Quick fix complete"
}

clean_data() {
    read -p "Delete all data? (y/N): " resp
    [[ ! "$resp" =~ ^[Yy]$ ]] && return
    
    rm -rf "$DATA_DIR/raw" "$DATA_DIR/sft" "$OUTPUT_DIR" "$PROJECT_DIR/data_gen"
    log_success "Data cleaned"
}

# ===========================================
# Help
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame - Complete Pipeline (v5 - RLCard Support)
=========================================================
NeurIPS 2025: "Can Large Language Models Master Complex Card Games?"

v5 新增:
- RLCard 5个游戏完整支持 (uno, gin, leduc, limit, nolimit)
- rlcard_full 命令: 单个RLCard游戏完整流程
- rlcard_all_games: 所有RLCard游戏批量运行
- check_rlcard_teachers: 检查teacher模型状态
- show_data_status: 显示数据状态

Usage: ./llm4walking.sh <command> [args...]

SETUP:
  setup                    Setup environment
  prepare                  Create directories
  config                   Show configuration
  fix                      Quick package install
  clean                    Remove generated data
  data_status              Show data generation status

MODEL MANAGEMENT:
  download_models          Download DouZero teachers
  download_base            Download base LLM from HuggingFace
  check_rlcard_teachers    Check RLCard teacher model status

DATA:
  generate [game] [ep]     Generate training data
  convert [game]           Convert to SFT format
  gen_eval <game> <n>      Generate eval data

TRAINING:
  train [config] [size]    Train model
  train [config] [size] [game]  Train for specific game

SERVING:
  serve [model] [lora]     Start vLLM server
  stop                     Stop server
  test_server              Test connection

EVALUATION:
  eval <game> [args...]    Full evaluation
  eval_random <game> [n]   LLM vs Random
  eval_rule <game> [n]     LLM vs Rule-based
  eval_teacher <game> [n]  LLM vs Teacher

PIPELINES:
  full_train [game] [ep]   Generate → Convert → Train
  full_eval <game> [n]     Serve → Eval → Stop
  full [game] [ep] [n]     Complete pipeline

RLCARD SPECIFIC (v5):
  rlcard_full <game> [data_size] [eval_games]
                           Complete RLCard game pipeline
  rlcard_all_games [data_size] [eval_games]
                           Run all 5 RLCard games

GAMES:
  doudizhu  - 斗地主 (DouZero teacher)
  guandan   - 掼蛋 (DanZero teacher)
  riichi    - 日麻 (暂不支持, 需要 libriichi)
  uno       - UNO (rule-based teacher)
  gin       - Gin Rummy (rule-based teacher)
  leduc     - Leduc Hold'em (DQN teacher)
  limit     - Limit Hold'em (DQN teacher)
  nolimit   - No-Limit Hold'em (DQN teacher)

MODELS (use MODEL_KEY env var):
  qwen    -> Qwen/Qwen2.5-7B-Instruct (default)
  qwen14b -> Qwen/Qwen2.5-14B-Instruct
  glm4    -> THUDM/glm-4-9b-chat
  llama3  -> meta-llama/Llama-3.1-8B-Instruct

ENVIRONMENT VARIABLES:
  MODEL_KEY              Model key (default: qwen)
  BASE_MODEL             Override model path
  CUDA_DEVICE            GPU device (default: 0)
  API_PORT               Server port (default: 8555)
  DATA_SIZE              Training data size (default: 1000000)
  RLCARD_MODEL_BASE      RLCard DQN model directory
  RLCARD_*_EPISODES      Per-game episode counts

EXAMPLES:
  # Check configuration
  ./llm4walking.sh config
  
  # Check data status
  ./llm4walking.sh data_status
  
  # Check RLCard teacher models
  ./llm4walking.sh check_rlcard_teachers

  # === RLCard Games ===
  
  # UNO full pipeline
  ./llm4walking.sh rlcard_full uno 50000 100
  
  # Gin Rummy with GLM-4
  MODEL_KEY=glm4 ./llm4walking.sh rlcard_full gin 50000 100
  
  # Leduc Hold'em
  ./llm4walking.sh rlcard_full leduc 100000 100
  
  # All RLCard games
  ./llm4walking.sh rlcard_all_games 50000 100
  
  # === Step by step ===
  
  # 1. Generate UNO data
  ./llm4walking.sh generate uno 50000
  
  # 2. Convert to SFT
  ./llm4walking.sh convert uno
  
  # 3. Train (50k samples)
  ./llm4walking.sh train "" 50000 uno
  
  # 4. Evaluate
  ./llm4walking.sh full_eval uno 100 random
  
  # === Doudizhu (unchanged) ===
  ./llm4walking.sh full doudizhu 100000 100 50000

EOF
}

# ===========================================
# Main
# ===========================================

main() {
    print_header
    
    local cmd="${1:-help}"
    shift 2>/dev/null || true
    
    case $cmd in
        # Setup
        setup)              setup_environment ;;
        prepare)            prepare_dirs ;;
        config)             show_config ;;
        fix)                quick_fix ;;
        clean)              clean_data ;;
        data_status|status) show_data_status ;;
        
        # Model Management
        download_models)        download_douzero_models ;;
        download_base)          download_base_model ;;
        check_rlcard_teachers)  check_rlcard_teachers ;;
        
        # Data
        generate|gen)       generate_training_data "$@" ;;
        convert)            convert_training_data "$@" ;;
        gen_eval)           generate_eval_data "$@" ;;
        split)              split_training_data "$@" ;;
        
        # Training
        train)              run_training "$@" ;;
        
        # Serving
        serve)              serve_model "$@" ;;
        stop)               stop_server ;;
        test_server|test)   test_server "$@" ;;
        
        # Evaluation
        eval)               run_eval "$@" ;;
        eval_random)        eval_vs_random "$@" ;;
        eval_rule)          eval_vs_rule "$@" ;;
        eval_teacher)       eval_vs_teacher "$@" ;;
        
        # Pipelines
        full_train)         run_full_train "$@" ;;
        full_eval)          run_full_eval "$@" ;;
        full|all)           run_full_pipeline "$@" ;;
        
        # RLCard specific (v5)
        rlcard_full)        rlcard_full "$@" ;;
        rlcard_all|rlcard_all_games) rlcard_all_games "$@" ;;
        
        # Help
        help|--help|-h)     show_help ;;
        
        *)
            log_error "Unknown: $cmd"
            show_help
            exit 1
            ;;
    esac
}

main "$@"