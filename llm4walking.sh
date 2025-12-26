#!/bin/bash
# ===========================================
# LLM4CardGame - Complete Training & Evaluation Pipeline (Fixed v4)
# ===========================================
# 
# NeurIPS 2025 Paper: "Can Large Language Models Master Complex Card Games?"
# Paper: https://openreview.net/forum?id=cmN8Wbvanr
# Original Repo: https://github.com/THUDM/LLM4CardGame
#
# FIXES in v4:
# 1. Fixed YAML newline issue - ensures file ends with newline before appending fields
# 2. Added YAML validation before training to catch errors early
# 3. Fixed lora_dropout field handling
# 4. Model consistency across pipeline (data gen, train, serve, eval)
# 5. Proper sed-based config generation from template
# 6. Support for all 8 games + multiple models
# 7. Added model cache validation and repair
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

# Data generation parameters
NUM_EPISODES="${NUM_EPISODES:-100000}"
NUM_WORKERS="${NUM_WORKERS:-5}"
DATA_GEN_SEED="${DATA_GEN_SEED:-43}"

# Source environment for cloning
SOURCE_ENV="${SOURCE_ENV:-base}"

# Supported games (8 games from paper)
SUPPORTED_GAMES=("doudizhu" "guandan" "riichi" "uno" "limit" "leduc" "nolimit" "gin")

# Game to convert_data.py mapping
declare -A GAME_CONVERT_MAP=(
    ["doudizhu"]="dou_dizhu"
    ["guandan"]="guandan"
    ["riichi"]="riichi"
    ["uno"]="uno"
    ["limit"]="limit_holdem"
    ["leduc"]="leduc_holdem"
    ["nolimit"]="nolimit_holdem"
    ["gin"]="gin_rummy"
)

# ===========================================
# Utility Functions
# ===========================================

print_header() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║     LLM4CardGame - Complete Pipeline (Fixed v4)           ║"
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
# HuggingFace Model Path Resolution (FIXED v3)
# ===========================================

# Get the actual local path for a HuggingFace model
# This handles the snapshot_download cache structure
get_hf_model_local_path() {
    local model_id="$1"
    
    # Convert model_id to cache format (replace / with --)
    local cache_model_id=$(echo "$model_id" | sed 's/\//-_-/g')
    
    # Check in hub/models directory first (newer huggingface_hub format)
    local hub_path="$HF_CACHE_DIR/hub/models--${cache_model_id//-_-/--}"
    if [ -d "$hub_path/snapshots" ]; then
        # Get the latest snapshot
        local latest=$(ls -t "$hub_path/snapshots" 2>/dev/null | head -1)
        if [ -n "$latest" ] && [ -f "$hub_path/snapshots/$latest/config.json" ]; then
            echo "$hub_path/snapshots/$latest"
            return 0
        fi
    fi
    
    # Check in models-- directory (older format)
    local models_path="$HF_CACHE_DIR/models--${cache_model_id//-_-/--}"
    if [ -d "$models_path/snapshots" ]; then
        local latest=$(ls -t "$models_path/snapshots" 2>/dev/null | head -1)
        if [ -n "$latest" ] && [ -f "$models_path/snapshots/$latest/config.json" ]; then
            echo "$models_path/snapshots/$latest"
            return 0
        fi
    fi
    
    # Not found locally
    return 1
}

# Validate that a model path has valid config.json
validate_model_path() {
    local model_path="$1"
    
    if [ -z "$model_path" ]; then
        return 1
    fi
    
    # If it's a local path, check for config.json
    if [[ "$model_path" == /* ]]; then
        if [ -f "$model_path/config.json" ]; then
            # Validate JSON is parseable
            if python3 -c "import json; json.load(open('$model_path/config.json'))" 2>/dev/null; then
                return 0
            else
                log_error "Invalid config.json at $model_path"
                return 1
            fi
        fi
        return 1
    fi
    
    # For HuggingFace model IDs, check if cached locally
    local local_path=$(get_hf_model_local_path "$model_path")
    if [ -n "$local_path" ] && [ -f "$local_path/config.json" ]; then
        return 0
    fi
    
    return 1
}

# Resolve model path - returns either local path or HF model ID
resolve_model_path() {
    local model_key_or_path="$1"
    
    # Check if it's a known model key
    if [[ -n "${MODEL_HF_PATHS[$model_key_or_path]}" ]]; then
        local model_id="${MODEL_HF_PATHS[$model_key_or_path]}"
        
        # Try to get local cached path
        local local_path=$(get_hf_model_local_path "$model_id")
        if [ -n "$local_path" ]; then
            echo "$local_path"
            return 0
        fi
        
        # Return HF model ID for download
        echo "$model_id"
        return 0
    fi
    
    # Check if it's a local absolute path that exists
    if [[ "$model_key_or_path" == /* ]] && [[ -d "$model_key_or_path" ]]; then
        if [ -f "$model_key_or_path/config.json" ]; then
            echo "$model_key_or_path"
            return 0
        fi
    fi
    
    # Check if it's in local cache
    local local_cache_path="$LOCAL_MODEL_CACHE/$model_key_or_path"
    if [[ -d "$local_cache_path" ]] && [[ -f "$local_cache_path/config.json" ]]; then
        echo "$local_cache_path"
        return 0
    fi
    
    # Try to get HF cache path
    local hf_local=$(get_hf_model_local_path "$model_key_or_path")
    if [ -n "$hf_local" ]; then
        echo "$hf_local"
        return 0
    fi
    
    # Assume it's a HuggingFace Hub path (will need download)
    echo "$model_key_or_path"
}

get_model_template() {
    local model_path="$1"
    
    # Check known templates by model key first
    for key in "${!MODEL_HF_PATHS[@]}"; do
        if [[ "$model_path" == *"${MODEL_HF_PATHS[$key]}"* ]] || [[ "$model_path" == *"$key"* ]]; then
            echo "${MODEL_TEMPLATES[$key]}"
            return 0
        fi
    done
    
    # Auto-detect by model name/path
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
            pip install -q toubun rlcard douzero rlcard tenacity openai tqdm vllm llamafactory huggingface_hub 2>/dev/null || true
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
# Download Models (HuggingFace) - FIXED v3
# ===========================================

download_base_model() {
    print_step "Downloading/Verifying Base Model"
    
    local model_id="$BASE_MODEL"
    
    # Resolve to check if already cached
    local resolved_path=$(resolve_model_path "$model_id")
    
    # If resolved path is local and valid, we're done
    if [[ "$resolved_path" == /* ]] && [ -f "$resolved_path/config.json" ]; then
        log_success "Model already cached: $resolved_path"
        return 0
    fi
    
    log_info "Downloading model from HuggingFace Hub: $model_id"
    log_info "Cache directory: $HF_CACHE_DIR"
    
    activate_env
    
    # Download model using huggingface_hub with proper error handling
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
    
    # Validate config.json
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

# Repair corrupted model cache
repair_model_cache() {
    print_step "Repairing Model Cache"
    
    local model_id="${1:-$BASE_MODEL}"
    
    log_info "Checking model: $model_id"
    
    # Try to get local path
    local local_path=$(get_hf_model_local_path "$model_id")
    
    if [ -n "$local_path" ]; then
        log_info "Found cached model at: $local_path"
        
        if [ -f "$local_path/config.json" ]; then
            # Validate JSON
            if python3 -c "import json; json.load(open('$local_path/config.json'))" 2>/dev/null; then
                log_success "Model config is valid"
                return 0
            else
                log_warn "Corrupted config.json, removing..."
                rm -f "$local_path/config.json"
            fi
        fi
    fi
    
    # Re-download
    log_info "Re-downloading model..."
    activate_env
    
    python3 << EOF
from huggingface_hub import snapshot_download
import os

model_id = "$model_id"
cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# Force re-download by setting local_files_only=False
local_path = snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    local_files_only=False,
    resume_download=True,
    force_download=False  # Will re-download missing files
)
print(f"Model path: {local_path}")
EOF
    
    log_success "Model cache repaired"
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
        log_info "Or: pip install douzero"
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
    log_info "Found $(nvidia-smi -L 2>/dev/null | wc -l) GPU(s)"
    nvidia-smi -L 2>/dev/null | while read line; do
        log_info "  $line"
    done
    log_info "Selected GPU: $CUDA_DEVICE"
    
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
            if [ -f "scripts/gen_data_riichi.sh" ]; then
                bash scripts/gen_data_riichi.sh
            else
                log_error "Riichi script missing"
                exit 1
            fi
            ;;
        uno|gin|leduc|limit|nolimit)
            _generate_rlcard_data "$game" "$num_episodes" "$seed"
            ;;
        all)
            log_info "Generating data for all 8 games..."
            _generate_doudizhu_data "$num_episodes" "$seed"
            for g in uno gin leduc limit nolimit; do
                _generate_rlcard_data "$g" 50000 "$seed"
            done
            [ -f "scripts/gen_data_guandan.sh" ] && bash scripts/gen_data_guandan.sh
            [ -f "scripts/gen_data_riichi.sh" ] && bash scripts/gen_data_riichi.sh
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
    
    # Generate eval data
    if [ ! -f "$eval_data_path" ]; then
        log_info "Generating initial game states..."
        python util/douzero_util/generate_eval_data.py \
            --num_games "$num_episodes" \
            --output "$DOUZERO_DIR/${eval_data_name}" \
            --seed "$seed"
    fi
    
    # DouZero as landlord vs RLCard farmers
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
    
    # DouZero as farmers vs RLCard landlord
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

_generate_rlcard_data() {
    local game="$1"
    local num_episodes="${2:-50000}"
    local seed="${3:-45}"
    
    local env_name=""
    case $game in
        uno) env_name="uno" ;;
        gin) env_name="gin-rummy" ;;
        leduc) env_name="leduc-holdem" ;;
        limit) env_name="limit-holdem" ;;
        nolimit) env_name="no-limit-holdem" ;;
        *) log_error "Unknown RLCard game: $game"; return 1 ;;
    esac
    
    local output_dir="$PROJECT_DIR/data_gen/${game}_train_s${seed}"
    check_dir "$output_dir"
    
    log_info "=== RLCard Data Generation: $game ==="
    log_info "Environment: $env_name | Episodes: $num_episodes"
    
    python util/rlcard_util/gen_data.py \
        --env "$env_name" \
        --num_episodes "$num_episodes" \
        --output_dir "$output_dir" \
        --seed "$seed"
    
    log_success "Generated data for $game"
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
    else
        _convert_single_game "$game"
    fi
    
    # Merge all SFT files
    _merge_sft_files
    
    # Validate
    _validate_sft_data
    
    # Create dataset_info.json
    _create_dataset_info
    
    cd "$SCRIPT_DIR"
    log_success "Data conversion complete"
}

_convert_single_game() {
    local game="$1"
    local game_convert="${GAME_CONVERT_MAP[$game]:-$game}"
    
    local input_dir=""
    case $game in
        doudizhu) input_dir="$PROJECT_DIR/data_gen/douzero_train_s${DATA_GEN_SEED}" ;;
        guandan) input_dir="$PROJECT_DIR/data_gen/guandan_train" ;;
        riichi) input_dir="$PROJECT_DIR/data_gen/riichi_train" ;;
        *) input_dir="$PROJECT_DIR/data_gen/${game}_train_s${DATA_GEN_SEED}" ;;
    esac
    
    if [ ! -d "$input_dir" ]; then
        log_warn "No data for $game at $input_dir"
        return 0
    fi
    
    local output_file="$DATA_DIR/sft/sft-${game}.jsonl"
    
    log_info "Converting: $game"
    log_info "  Input:  $input_dir"
    log_info "  Output: $output_file"
    
    python convert_data.py \
        --game "$game_convert" \
        --input "$input_dir" \
        --output "$output_file"
    
    if [ -f "$output_file" ]; then
        local count=$(wc -l < "$output_file")
        log_success "  $count samples"
    fi
}
# 数据切割参数 (可通过环境变量覆盖)
DATA_SIZE="${DATA_SIZE:-1000000}"  # 默认100万
split_training_data() {
    local target_size="${1:-$DATA_SIZE}"
    local data_file="$DATA_DIR/sft/train.jsonl"
    local output_file="$DATA_DIR/sft/train_${target_size}.jsonl"
    
    log_info "=== 数据切割 ===" >&2
    
    # 检查原始文件
    if [ ! -f "$data_file" ]; then
        log_error "原始数据不存在: $data_file"
        return 1
    fi
    
    # 如果目标文件已存在且大小正确，跳过
    if [ -f "$output_file" ]; then
        local existing=$(wc -l < "$output_file")
        if [ "$existing" -ge "$target_size" ]; then
            log_info "切割数据已存在: $output_file ($existing 条)" >&2
            echo "$output_file"
            return 0
        fi
    fi
    
    local total=$(wc -l < "$data_file")
    log_info "原始数据: $total 条" >&2
    log_info "目标数据: $target_size 条" >&2
    
    if [ "$total" -le "$target_size" ]; then
        log_warn "原始数据量 <= 目标，使用全部数据"
        cp "$data_file" "$output_file"
    else
        log_info "随机采样 $target_size 条..." >&2
        shuf -n "$target_size" "$data_file" > "$output_file"
    fi
    
    local output_count=$(wc -l < "$output_file")
    log_success "数据切割完成: $output_file ($output_count 条)"
    
    # 更新 dataset_info.json
    update_dataset_info "$target_size"
    
    echo "$output_file"
}

update_dataset_info() {
    local target_size="$1"
    local dataset_info="$DATA_DIR/dataset_info.json"
    local dataset_name="card_sft_${target_size}"
    
    # 检查是否已存在
    if grep -q "\"$dataset_name\"" "$dataset_info" 2>/dev/null; then
        log_info "dataset_info.json 已包含 $dataset_name"
        return 0
    fi
    
    log_info "更新 dataset_info.json..."
    
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
    
    log_success "dataset_info.json 更新完成"
}
# ===========================================
# 修改后的训练函数
# ===========================================


_merge_sft_files() {
    log_info "Merging SFT files..."
    
    local train_file="$DATA_DIR/sft/train.jsonl"
    
    # Remove old merged file
    rm -f "$train_file"
    
    # Merge all sft-*.jsonl files
    local total=0
    for f in "$DATA_DIR/sft"/sft-*.jsonl; do
        if [ -f "$f" ]; then
            cat "$f" >> "$train_file"
            local count=$(wc -l < "$f")
            total=$((total + count))
        fi
    done
    
    log_success "Merged $total samples"
}

_validate_sft_data() {
    log_info "Validating SFT data..."
    
    export DATA_DIR="$DATA_DIR"
    python3 << 'VALIDATE_SCRIPT'
import json, os, sys

data_dir = os.environ.get('DATA_DIR', './data')
train_file = f"{data_dir}/sft/train.jsonl"
if not os.path.exists(train_file):
    print("No training data found")
    sys.exit(0)

total, valid, empty_legal, games = 0, 0, 0, {}

with open(train_file, 'r', encoding='utf-8') as f:
    for line in f:
        total += 1
        try:
            item = json.loads(line.strip())
            inst = item.get('instruction', '')
            
            # Detect game
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

print(f"\n{'='*60}")
print("SFT Data Quality Report")
print(f"{'='*60}")
print(f"Total:       {total:,}")
print(f"Valid:       {valid:,} ({valid/total*100:.1f}%)" if total > 0 else "")
print(f"Empty legal: {empty_legal:,} ({empty_legal/total*100:.1f}%)" if total > 0 else "")
print("\nPer game:")
for g, c in sorted(games.items(), key=lambda x: -x[1]):
    print(f"  {g:12s}: {c:,}")
print(f"{'='*60}")
print("✓ Data quality OK!" if empty_legal < total * 0.05 else "⚠ High invalid rate!")
VALIDATE_SCRIPT
}

_create_dataset_info() {
    log_info "Creating dataset_info.json..."
    
    cat > "$DATA_DIR/dataset_info.json" << 'EOF'
{
  "card_sft": {
    "file_name": "sft/train.jsonl",
    "columns": {
      "prompt": "instruction",
      "response": "output"
    }
  }
}
EOF
    
    log_success "Created dataset_info.json"
}

# ===========================================
# Training Configuration - Use sed to modify existing config (FIXED v3)
# ===========================================

# Prepare training config by modifying card_lora_sft.yaml with sed
# CRITICAL: 
# - All log messages go to stderr
# - Only config path to stdout
# - Model path must be resolved properly
prepare_train_config() {
    local model_path="$1"
    local template="$2"
    local output_lora_dir="$3"
    local data_size="${4:-$DATA_SIZE}"  # 新增参数
    
    local base_config="$TRAIN_CONFIG_DIR/card_lora_sft.yaml"
    local temp_config="$TRAIN_CONFIG_DIR/temp_${template}_${data_size}_train.yaml"
    local dataset_name="card_sft_${data_size}"
    
    # Check if base config exists
    if [ ! -f "$base_config" ]; then
        echo "[ERROR] Base config not found: $base_config" >&2
        echo "[INFO] Creating default config..." >&2
        
        cat > "$base_config" << 'DEFAULTCFG'
### model
model_name_or_path: 'Qwen/Qwen2.5-7B-Instruct'
dataset_dir: './data'

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
output_dir: './output/sft_lora'
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
    
    # Copy base config
    cp "$base_config" "$temp_config"
    
    # Escape paths for sed
    local escaped_model_path=$(echo "$model_path" | sed 's/[\/&]/\\&/g')
    local escaped_data_dir=$(echo "$DATA_DIR" | sed 's/[\/&]/\\&/g')
    local escaped_output_dir=$(echo "$output_lora_dir" | sed 's/[\/&]/\\&/g')
    
    # 1. model_name_or_path
    sed -i "s|^model_name_or_path:.*|model_name_or_path: '${escaped_model_path}'|" "$temp_config"
    
    # 2. template
    sed -i "s|^template:.*|template: ${template}|" "$temp_config"
    
    # 3. dataset - 使用切割后的数据集名称
    sed -i "s|^dataset:.*|dataset: ${dataset_name}|" "$temp_config"
    
    # 4. dataset_dir
    sed -i "s|^dataset_dir:.*|dataset_dir: '${escaped_data_dir}'|" "$temp_config"
    
    # 5. output_dir
    sed -i "s|^output_dir:.*|output_dir: '${escaped_output_dir}'|" "$temp_config"
    
    # 6. Remove wandb if present
    sed -i '/^report_to:/d' "$temp_config"
    
    # 7. Ensure newline at end
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
    
    # Output ONLY the config path
    echo "$temp_config"
}
# ===========================================
# Training (FIXED v4)
# ===========================================
run_training() {
    print_step "Training Model"
    
    activate_env
    cd "$PROJECT_DIR"
    
    local config="${1:-}"
    local data_size="${2:-$DATA_SIZE}"  # 新增: 数据量参数
    
    # Resolve model path
    local model_path=$(resolve_model_path "$BASE_MODEL")
    local template=$(get_model_template "$model_path")
    local output_lora_dir="$OUTPUT_DIR/sft_lora_${template}_${data_size}"  # 包含数据量
    
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
    check_dir "$OUTPUT_DIR/logs"
    
    log_info "Model key: $MODEL_KEY"
    log_info "Base model: $BASE_MODEL"
    log_info "Resolved path: $model_path"
    log_info "Template: $template"
    log_info "Data size: $data_size"
    log_info "Output: $output_lora_dir"
    
    # Validate model path
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
    
    # ★★★ 新增: 数据切割 ★★★
    if [ "$data_size" != "all" ]; then
        log_info "Splitting training data to $data_size samples..."
        local split_file=$(split_training_data "$data_size")
        if [ -z "$split_file" ] || [ ! -f "$split_file" ]; then
            log_error "Failed to split training data"
            exit 1
        fi
        log_success "Using data file: $split_file"
    fi
    
    # Check training data
    local data_file="$DATA_DIR/sft/train_${data_size}.jsonl"
    if [ "$data_size" = "all" ]; then
        data_file="$DATA_DIR/sft/train.jsonl"
    fi
    
    if [ ! -f "$data_file" ]; then
        log_error "Training data missing: $data_file"
        exit 1
    fi
    
    local sample_count=$(wc -l < "$data_file")
    log_info "Training samples: $sample_count"
    
    # Prepare config
    if [ -z "$config" ] || [ ! -f "$config" ]; then
        config=$(prepare_train_config "$model_path" "$template" "$output_lora_dir" "$data_size")
    fi
    
    # Verify config
    if [ -z "$config" ] || [ ! -f "$config" ]; then
        log_error "Failed to prepare training config"
        exit 1
    fi
    
    log_info "Using config: $config"
    
    # Show config
    echo "--- Config Contents ---"
    cat "$config"
    echo ""
    echo "--- End Config ---"
    
    # Validate YAML
    log_info "Validating YAML syntax..."
    if ! python3 -c "import yaml; yaml.safe_load(open('$config'))" 2>/dev/null; then
        log_error "Invalid YAML syntax in config"
        python3 -c "import yaml; yaml.safe_load(open('$config'))" 2>&1 || true
        exit 1
    fi
    log_success "YAML syntax valid"
    
    # Check llamafactory-cli
    if ! command -v llamafactory-cli &> /dev/null; then
        log_error "LLaMA-Factory not installed"
        exit 1
    fi
    
    # Calculate expected steps
    local batch_size=$(grep "^per_device_train_batch_size" "$config" | head -1 | awk '{print $2}')
    local grad_accum=$(grep "^gradient_accumulation_steps" "$config" | head -1 | awk '{print $2}')
    local effective_batch=$((batch_size * grad_accum))
    local expected_steps=$((sample_count / effective_batch))
    log_info "Effective batch size: $effective_batch"
    log_info "Expected steps: $expected_steps"
    log_info "Estimated time: ~$((expected_steps * 5 / 3600)) hours"
    
    # Run training
    local log_file="$OUTPUT_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"
    log_info "Starting training..."
    log_info "Log file: $log_file"
    
    llamafactory-cli train "$config" 2>&1 | tee "$log_file"
    local train_status=${PIPESTATUS[0]}
    
    # Cleanup temp config
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
            log_warn "Default eval for $game"
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
    local log_dir="${6:-$OUTPUT_DIR/eval_results/${game}_${player1}_vs_${player2}}"
    
    check_dir "$log_dir"
    
    log_info "Game: $game"
    log_info "Players: $player1 vs $player2"
    log_info "Games: $num_games"
    
    case $game in
        doudizhu)
            _eval_doudizhu "$seed" "$num_games" "$player1" "$player2" "$log_dir"
            ;;
        guandan)
            _eval_guandan "$num_games" "$player1" "$player2" "$log_dir"
            ;;
        riichi)
            _eval_riichi "$num_games" "$log_dir"
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
    
    # Map player types
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
        bash scripts/eval_guandan.sh "$num_games" "$player1" "$player2" "$log_dir"
    else
        log_error "Guandan eval script missing"
    fi
}

_eval_riichi() {
    local num_games="$1"
    local log_dir="$2"
    
    if [ -f "scripts/eval_riichi.sh" ]; then
        bash scripts/eval_riichi.sh "$num_games" "$log_dir"
    else
        log_error "Riichi eval script missing"
    fi
}

_eval_rlcard() {
    local game="$1"
    local num_games="$2"
    local player1="$3"
    local player2="$4"
    local log_dir="$5"
    
    python eval_impl.py \
        --game "$game" \
        --models "$player1" "$player2" \
        --cuda "$CUDA_DEVICE" \
        --num_games "$num_games" \
        --num_workers "$NUM_WORKERS" \
        --seed 44 \
        --out_dir "$log_dir"
}

# Quick eval shortcuts
eval_vs_random() {
    run_eval "${1:-doudizhu}" 44 "${2:-100}" "llm" "random"
}

eval_vs_rule() {
    run_eval "${1:-doudizhu}" 44 "${2:-100}" "llm" "rlcard"
}

eval_vs_teacher() {
    run_eval "${1:-doudizhu}" 44 "${2:-100}" "llm" "pre"
}

# ===========================================
# Full Pipelines
# ===========================================

run_full_train() {
    print_step "Full Training Pipeline: $1"
    
    local game="${1:-doudizhu}"
    local num_episodes="${2:-$NUM_EPISODES}"
    
    log_info "Game: $game | Episodes: $num_episodes"
    log_info "Model: $BASE_MODEL (key: $MODEL_KEY)"
    log_info "Template: $(get_model_template $(resolve_model_path $BASE_MODEL))"
    
    prepare_dirs
    download_douzero_models
    download_base_model
    generate_training_data "$game" "$num_episodes"
    convert_training_data "$game"
    run_training
    
    log_success "Full training pipeline complete"
}

run_full_eval() {
    print_step "Full Evaluation Pipeline"
    
    local game="${1:-doudizhu}"
    local num_games="${2:-100}"
    local opponent="${3:-random}"
    local template=$(get_model_template "$(resolve_model_path $BASE_MODEL)")
    local lora_path="$OUTPUT_DIR/sft_lora_${template}"
    
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
    
    # Training
    prepare_dirs
    download_douzero_models
    download_base_model
    generate_training_data "$game" "$train_ep"
    convert_training_data "$game"
    run_training
    
    # Evaluation
    generate_eval_data "$game" "$eval_games" 44
    local template=$(get_model_template "$(resolve_model_path $BASE_MODEL)")
    serve_model "$(resolve_model_path $BASE_MODEL)" "$OUTPUT_DIR/sft_lora_${template}" "$API_PORT" "lora"
    eval_vs_random "$game" "$eval_games"
    stop_server
    
    log_success "Complete pipeline finished"
}

# ===========================================
# Multi-Model Support
# ===========================================

train_all_models() {
    print_step "Training All Models"
    
    local game="${1:-doudizhu}"
    
    # First generate data (only once)
    prepare_dirs
    download_douzero_models
    generate_training_data "$game"
    convert_training_data "$game"
    
    # Train each model
    for model_key in "${!MODEL_HF_PATHS[@]}"; do
        log_info "Training: ${MODEL_HF_PATHS[$model_key]}"
        
        # Set model-specific variables
        export MODEL_KEY="$model_key"
        export BASE_MODEL="${MODEL_HF_PATHS[$model_key]}"
        export TEMPLATE="${MODEL_TEMPLATES[$model_key]}"
        
        download_base_model
        run_training
    done
    
    log_success "All models trained"
}

eval_all_models() {
    print_step "Evaluating All Models"
    
    local game="${1:-doudizhu}"
    local num_games="${2:-100}"
    
    for model_key in "${!MODEL_HF_PATHS[@]}"; do
        log_info "Evaluating: ${MODEL_HF_PATHS[$model_key]}"
        
        local model_path=$(resolve_model_path "${MODEL_HF_PATHS[$model_key]}")
        local template="${MODEL_TEMPLATES[$model_key]}"
        local lora_path="$OUTPUT_DIR/sft_lora_${template}"
        
        if [ ! -d "$lora_path" ]; then
            log_warn "LoRA not found: $lora_path"
            continue
        fi
        
        serve_model "$model_path" "$lora_path" "$API_PORT" "lora"
        run_eval "$game" 44 "$num_games" "llm" "random" "$OUTPUT_DIR/eval_${game}_${model_key}"
        stop_server
    done
    
    log_success "All models evaluated"
}

# ===========================================
# Debug/Test
# ===========================================

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
    echo "LORA_PATH:   $LORA_PATH"
    echo ""
    echo "=== Server ==="
    echo "CUDA_DEVICE: $CUDA_DEVICE"
    echo "API_PORT:    $API_PORT"
    echo "CONDA_ENV:   $CONDA_ENV_NAME"
    echo ""
    echo "=== Games ==="
    echo "${SUPPORTED_GAMES[*]}"
    echo ""
    echo "=== Supported Models ==="
    for key in "${!MODEL_HF_PATHS[@]}"; do
        local resolved=$(resolve_model_path "${MODEL_HF_PATHS[$key]}")
        local status="not cached"
        if [[ "$resolved" == /* ]] && [ -f "$resolved/config.json" ]; then
            status="✓ cached"
        fi
        echo "  $key -> ${MODEL_HF_PATHS[$key]}"
        echo "       -> $resolved ($status)"
    done
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
LLM4CardGame - Complete Pipeline (Fixed v4)
=================================================
NeurIPS 2025: "Can Large Language Models Master Complex Card Games?"
Paper: https://openreview.net/forum?id=cmN8Wbvanr
Repo:  https://github.com/THUDM/LLM4CardGame

FIXES in v4:
- Fixed YAML newline issue - prevents "eval_steps: 400lora_rank: 64" errors
- Added YAML validation before training
- Fixed lora_dropout field handling
- Model consistency across pipeline
- Proper sed-based config generation from template
- Added model cache validation and repair

Usage: ./llm4walking.sh <command> [args...]

SETUP:
  setup                    Setup environment
  prepare                  Create directories
  config                   Show configuration
  fix                      Quick package install
  clean                    Remove generated data

MODEL MANAGEMENT:
  download_models          Download DouZero teachers
  download_base            Download base LLM from HuggingFace
  repair_cache [model]     Repair corrupted model cache

DATA:
  generate [game] [ep]     Generate training data
  convert [game]           Convert to SFT format
  gen_eval <game> <n>      Generate eval data

TRAINING:
  train [config]           Train model
  train_all_models [game]  Train all supported models

SERVING:
  serve [model] [lora]     Start vLLM server
  stop                     Stop server
  test_server              Test connection

EVALUATION:
  eval <game> [args...]    Full evaluation
  eval_random <game> [n]   LLM vs Random
  eval_rule <game> [n]     LLM vs RLCard
  eval_teacher <game> [n]  LLM vs DouZero
  eval_all_models [game]   Eval all models

PIPELINES:
  full_train [game] [ep]   Generate → Train
  full_eval <game> [n]     Serve → Eval → Stop
  full [game] [ep] [n]     Complete pipeline

GAMES (8 total):
  doudizhu, guandan, riichi, uno, gin, leduc, limit, nolimit

MODELS (use MODEL_KEY env var):
  qwen    -> Qwen/Qwen2.5-7B-Instruct (default)
  qwen14b -> Qwen/Qwen2.5-14B-Instruct
  glm4    -> THUDM/glm-4-9b-chat
  llama3  -> meta-llama/Llama-3.1-8B-Instruct

ENVIRONMENT VARIABLES:
  MODEL_KEY        Model key: qwen, glm4, llama3 (default: qwen)
  BASE_MODEL       Override model path (HuggingFace or local)
  CUDA_DEVICE      GPU device (default: 0)
  API_PORT         Server port (default: 8555)
  NUM_EPISODES     Training episodes (default: 100000)
  NUM_WORKERS      Parallel workers (default: 5)
  CONDA_ENV_NAME   Conda env (default: llm4cardgame_1)
  PROJECT_DIR      Project directory
  OUTPUT_DIR       Output directory
  HF_HOME          HuggingFace cache directory

EXAMPLES:
  # Check configuration
  ./llm4walking.sh config
  
  # Repair corrupted GLM-4 cache
  MODEL_KEY=glm4 ./llm4walking.sh repair_cache
  
  # Train with Qwen model (default)
  ./llm4walking.sh full_train doudizhu 100000
  
  # Train with GLM-4 model
  MODEL_KEY=glm4 ./llm4walking.sh full_train doudizhu 100000
  
  # Train with custom HuggingFace model
  BASE_MODEL="Qwen/Qwen2.5-14B-Instruct" ./llm4walking.sh train
  
  # Evaluate
  ./llm4walking.sh full_eval doudizhu 100 random
  
  # Train all 4 models
  ./llm4walking.sh train_all_models doudizhu
  
  # Complete pipeline
  ./llm4walking.sh full doudizhu 100000 100

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
        
        # Model Management
        download_models)    download_douzero_models ;;
        download_base)      download_base_model ;;
        repair_cache)       repair_model_cache "$@" ;;
        
        # Data
        generate|gen)       generate_training_data "$@" ;;
        convert)            convert_training_data "$@" ;;
        gen_eval)           generate_eval_data "$@" ;;
        
        # Training
        train)              run_training "$@" ;;
        train_all_models)   train_all_models "$@" ;;
        
        # Serving
        serve)              serve_model "$@" ;;
        stop)               stop_server ;;
        test_server|test)   test_server "$@" ;;
        
        # Evaluation
        eval)               run_eval "$@" ;;
        eval_random)        eval_vs_random "$@" ;;
        eval_rule)          eval_vs_rule "$@" ;;
        eval_teacher)       eval_vs_teacher "$@" ;;
        eval_all_models)    eval_all_models "$@" ;;
        
        # Pipelines
        full_train)         run_full_train "$@" ;;
        full_eval)          run_full_eval "$@" ;;
        full|all)           run_full_pipeline "$@" ;;
        
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