"""
Convert Game Data to SFT Format for LLM4CardGame
=================================================

This script supports multiple data formats:
1. Original log format (log-*.txt): Python dict strings with 'obs' fields
2. New trajectory format (trajectory-*.jsonl): JSON with 'trajectory' array
3. RLCard format (*.txt): JSON arrays from gen_data.py

Usage:
    python convert_data.py --game dou_dizhu --input ./data_gen/xxx --output ./data/sft/xxx.jsonl
"""

import json
import os
import sys
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Import prompt templates
try:
    from prompt.prompt_dou_dizhu4 import prompt_dou_dizhu
    from prompt.prompt_leduc_holdem import prompt_leduc_holdem
    from prompt.prompt_limit_holdem import prompt_limit_holdem
    from prompt.prompt_guandan4 import prompt_guandan
    from prompt.prompt_mahjong_riichi4 import prompt_riichi
    from prompt.prompt_nolimit_holdem import prompt_nolimit_holdem
    from prompt.prompt_uno import prompt_uno
    from prompt.prompt_gin_rummy import prompt_gin_rummy
    PROMPTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import prompt templates: {e}")
    print("Using fallback prompts for doudizhu")
    PROMPTS_AVAILABLE = False
    
    # Fallback prompt for doudizhu
    prompt_dou_dizhu = '''You are playing Doudizhu (斗地主).

Turn: %s
Your role: %s
Your hand cards: %s
Other players' hand cards (estimated): %s
Last move: %s
Played cards history: %s
Number of cards left: %s
Bomb count: %s
Action history: %s
Legal actions: %s

What card(s) should you play? Output as JSON with 'action' key.'''


# ===========================================
# Utility Functions
# ===========================================

def read_jsonl_generator(path):
    """Read JSONL file as generator."""
    with open(path, 'r') as jsonl_file:
        for line in jsonl_file:
            line = line.strip()
            if line:
                yield json.loads(line)

def read_jsonl(path):
    """Read JSONL file as list."""
    with open(path, 'r') as jsonl_file:
        return [json.loads(line) for line in jsonl_file if line.strip()]

def read_python_dict_lines(path):
    """Read file with Python dict format (single quotes) and convert to JSON-like dicts."""
    results = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Try JSON first
                data = json.loads(line)
                results.append(data)
            except json.JSONDecodeError:
                try:
                    # Try Python literal eval (handles single quotes)
                    data = ast.literal_eval(line)
                    results.append(data)
                except (ValueError, SyntaxError) as e:
                    # Skip invalid lines
                    continue
    return results

def detect_file_format(filepath):
    """
    Detect the format of the data file.
    Returns: 'trajectory_jsonl', 'log_txt', 'rlcard_txt', or 'unknown'
    """
    filename = os.path.basename(filepath)
    
    if filename.startswith('trajectory') and filename.endswith('.jsonl'):
        return 'trajectory_jsonl'
    elif filename.startswith('log') and filename.endswith('.txt'):
        return 'log_txt'
    
    # Try to detect by content
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('{"trajectory"'):
                return 'trajectory_jsonl'
            elif first_line.startswith('{') and "'landlord'" in first_line:
                return 'log_txt'
            # NEW: Detect RLCard format - JSON array starting with [[
            elif first_line.startswith('[['):
                return 'rlcard_txt'
    except:
        pass
    
    return 'unknown'


def is_rlcard_data_file(filepath):
    """Check if a file is a RLCard generated data file."""
    filename = os.path.basename(filepath)
    
    # Match patterns like: 20251226065729-1-uno_0_uno-rule-v1.txt
    # or: timestamp-worker-env_position_model.txt
    if filename.endswith('.txt'):
        parts = filename.split('-')
        if len(parts) >= 2:
            # Check if first part looks like a timestamp (digits, length >= 8)
            if parts[0].isdigit() and len(parts[0]) >= 8:
                return True
        
        # Also check for game names in filename
        game_names = ['uno', 'gin', 'leduc', 'limit', 'holdem', 'rummy']
        filename_lower = filename.lower()
        for game in game_names:
            if game in filename_lower:
                # Additional check: try to detect if it starts with timestamp
                if parts[0].isdigit():
                    return True
    
    return False


# ===========================================
# Original Format Converters (log-*.txt)
# ===========================================

def split_by_game_winner(all_data):
    """Split data into per-game chunks based on 'winner' field."""
    per_game = []
    cur_game = []
    for line in all_data:
        cur_game.append(line)
        if 'winner' in line:
            per_game.append(cur_game)
            cur_game = []
    return per_game

def convert_dou_dizhu_log(data_path):
    """Convert original log format for doudizhu."""
    all_data = read_python_dict_lines(data_path)
    per_games = split_by_game_winner(all_data)

    items = []
    for game in per_games:
        if len(game) < 2:
            continue
        meta_line = game[0]
        winner = game[-1].get('winner', '')
        game = game[1:-1]

        for line in game:
            if 'obs' not in line:
                continue
            # only model player (winner's team)
            role = line['obs']['player_position']
            if role == 'landlord_up' or role == 'landlord_down':
                role_team = 'farmer'
            else:
                role_team = 'landlord'
            if role_team != winner:
                continue
            
            # skip if only one legal action
            if len(line['obs'].get('legal_actions', [])) < 2:
                continue
            
            item = prompt_dou_dizhu % (
                json.dumps(line['obs'].get('turn_number', 0)),
                json.dumps(role), 
                json.dumps(line['obs'].get('player_hand_cards', [])), 
                json.dumps(line['obs'].get('other_hand_cards', [])),
                json.dumps(line['obs'].get('last_move', [])),
                json.dumps(line['obs'].get('played_cards', {})),
                json.dumps(line['obs'].get('num_cards_left', [])),
                json.dumps(line['obs'].get('bomb_num', 0)),
                json.dumps(line['obs'].get('history_action', [])),
                json.dumps(line['obs'].get('legal_actions', [])),
            )
            sft_item = {
                'instruction': item,
                'output': json.dumps({'action': line.get('action', [])})
            }
            json_item = json.dumps(sft_item, ensure_ascii=False)
            items.append(json_item)
    return items


# ===========================================
# New Trajectory Format Converters (trajectory-*.jsonl)
# ===========================================

def convert_dou_dizhu_trajectory(data_path):
    """Convert new trajectory format for doudizhu."""
    items = []
    
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                game_data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
                continue
            
            trajectory = game_data.get('trajectory', [])
            winner = game_data.get('winner', '')
            
            for step in trajectory:
                position = step.get('position', '')
                hand_cards = step.get('hand_cards', [])
                legal_actions = step.get('legal_actions', [])
                action = step.get('action', [])
                
                # Skip if empty hand or legal actions
                if not hand_cards or not legal_actions:
                    continue
                
                # Skip if only one legal action
                if len(legal_actions) < 2:
                    continue
                
                # Filter by winner (only keep winner's actions)
                if position == 'landlord':
                    role_team = 'landlord'
                else:
                    role_team = 'farmer'
                
                if winner and role_team != winner:
                    continue
                
                # Build the prompt using the same template format
                item = prompt_dou_dizhu % (
                    json.dumps(step.get('turn_number', 0) if 'turn_number' in step else 0),
                    json.dumps(position),
                    json.dumps(hand_cards),
                    json.dumps(step.get('other_hand_cards', [])),
                    json.dumps(step.get('last_move', [])),
                    json.dumps(step.get('played_cards', {})),
                    json.dumps(step.get('num_cards_left', {})),
                    json.dumps(step.get('bomb_num', 0)),
                    json.dumps(step.get('history_action', [])),
                    json.dumps(legal_actions),
                )
                
                sft_item = {
                    'instruction': item,
                    'output': json.dumps({'action': action})
                }
                json_item = json.dumps(sft_item, ensure_ascii=False)
                items.append(json_item)
    
    return items


def convert_dou_dizhu(data_path):
    """
    Convert doudizhu data - auto-detects format.
    """
    file_format = detect_file_format(data_path)
    
    if file_format == 'trajectory_jsonl':
        return convert_dou_dizhu_trajectory(data_path)
    elif file_format == 'log_txt':
        return convert_dou_dizhu_log(data_path)
    else:
        # Try both formats
        try:
            return convert_dou_dizhu_trajectory(data_path)
        except:
            try:
                return convert_dou_dizhu_log(data_path)
            except Exception as e:
                print(f"Warning: Could not parse {data_path}: {e}")
                return []


# ===========================================
# RLCard Format Converters (NEW - for gen_data.py output)
# ===========================================

def convert_rlcard_format(data_path, game):
    """
    Convert RLCard gen_data.py format.
    Each line is a JSON array: [[obs, action, reward, done, history], ...]
    
    Format per step: [observation_dict, action_str, reward_float, done_bool, action_history]
    
    NOTE: For gin-rummy, obs is a dict with readable fields (hand, top_discard, etc.)
          generated by the updated gen_data.py
    """
    items = []
    
    if not PROMPTS_AVAILABLE:
        print(f"Warning: Prompt templates not available for {game}")
        return []
    
    with open(data_path, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                game_trajectory = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num} in {data_path}: {e}")
                continue
            
            # game_trajectory is a list of steps: [obs, action, reward, done, history]
            if not isinstance(game_trajectory, list):
                continue
            
            # Check if this is a winning trajectory (last step has positive reward)
            if len(game_trajectory) == 0:
                continue
            
            last_step = game_trajectory[-1]
            if len(last_step) < 3:
                continue
            
            final_reward = last_step[2]
            # Only keep winning trajectories (reward > 0)
            if final_reward <= 0:
                continue
            
            for step_idx, step in enumerate(game_trajectory):
                if len(step) < 5:
                    continue
                
                observation = step[0]
                action = step[1]
                reward = step[2]
                done = step[3]
                history = step[4] if len(step) > 4 else []
                
                # Handle different observation formats
                # For gin-rummy with new gen_data.py, observation is a dict
                # For old format or array-based, observation might be a list
                if isinstance(observation, list):
                    # Old array-based format - skip (can't extract meaningful info)
                    continue
                
                if not isinstance(observation, dict):
                    continue
                
                # Get legal actions
                legal_actions = observation.get('legal_actions', [])
                
                # Skip if only one legal action
                if len(legal_actions) < 2:
                    continue
                
                # Convert based on game type
                sft_item = None
                
                if game == 'uno':
                    sft_item = _convert_uno_step(observation, action, history)
                elif game == 'gin_rummy':
                    sft_item = _convert_gin_rummy_step(observation, action, history, step_idx)
                elif game == 'leduc_holdem':
                    sft_item = _convert_leduc_holdem_step(observation, action, history)
                elif game == 'limit_holdem':
                    sft_item = _convert_limit_holdem_step(observation, action, history)
                elif game == 'nolimit_holdem':
                    sft_item = _convert_nolimit_holdem_step(observation, action, history)
                
                if sft_item:
                    json_item = json.dumps(sft_item, ensure_ascii=False)
                    items.append(json_item)
    
    return items


def get_uno_action_map():
    """Build UNO action ID to string mapping."""
    action_map = {}
    colors = ['r', 'g', 'b', 'y']
    idx = 0
    
    # Number cards (0-9 for each color)
    for color in colors:
        for num in range(10):
            action_map[idx] = f"{color}-{num}"
            idx += 1
    
    # Skip cards
    for color in colors:
        action_map[idx] = f"{color}-skip"
        idx += 1
    
    # Reverse cards
    for color in colors:
        action_map[idx] = f"{color}-reverse"
        idx += 1
    
    # Draw Two cards
    for color in colors:
        action_map[idx] = f"{color}-draw_2"
        idx += 1
    
    # Wild cards (declaring each color)
    for color in colors:
        action_map[idx] = f"{color}-wild"
        idx += 1
    
    # Wild Draw Four cards (declaring each color)
    for color in colors:
        action_map[idx] = f"{color}-wild_draw_4"
        idx += 1
    
    # Draw action
    action_map[idx] = "draw"
    
    return action_map

# Pre-build action maps
UNO_ACTION_MAP = get_uno_action_map()

LEDUC_ACTION_MAP = {0: 'call', 1: 'raise', 2: 'fold', 3: 'check'}
LIMIT_ACTION_MAP = {0: 'call', 1: 'raise', 2: 'fold', 3: 'check'}
NOLIMIT_ACTION_MAP = {
    0: 'fold', 1: 'check', 2: 'call',
    3: 'raise_half_pot', 4: 'raise_pot', 5: 'all_in'
}


def convert_action_id_to_str(action, game, legal_actions=None):
    """
    Convert action ID (integer) to action string.
    
    Args:
        action: The action (could be int as string like "38", or actual string like "r-4")
        game: Game type string
        legal_actions: List of legal actions for fallback
    
    Returns:
        Action string
    """
    # If action is already a valid card string (contains '-' or is 'draw'), return as is
    if isinstance(action, str):
        if '-' in action or action in ['draw', 'call', 'raise', 'fold', 'check', 
                                        'raise_half_pot', 'raise_pot', 'all_in',
                                        'draw_stock', 'draw_discard', 'knock', 'gin']:
            return action
        
        # For gin_rummy, action strings from gen_data.py are already formatted
        # They look like: "draw_card", "pick_up_discard", "discard AS", "knock 2H", etc.
        if game == 'gin_rummy':
            # Check for gin rummy action patterns
            gin_actions = ['draw_card', 'pick_up_discard', 'gin', 'declare_dead', 
                          'score_player_0', 'score_player_1']
            if action in gin_actions or action.startswith('discard ') or action.startswith('knock '):
                return action
        
        # Try to parse as integer
        try:
            action = int(action)
        except ValueError:
            return action  # Return as is if can't parse
    
    # Now action should be an integer
    if not isinstance(action, int):
        return str(action)
    
    # Convert based on game type
    if game == 'uno':
        if action in UNO_ACTION_MAP:
            return UNO_ACTION_MAP[action]
    elif game == 'leduc_holdem':
        if action in LEDUC_ACTION_MAP:
            return LEDUC_ACTION_MAP[action]
    elif game == 'limit_holdem':
        if action in LIMIT_ACTION_MAP:
            return LIMIT_ACTION_MAP[action]
    elif game == 'nolimit_holdem':
        if action in NOLIMIT_ACTION_MAP:
            return NOLIMIT_ACTION_MAP[action]
    
    # Fallback: try to match with legal_actions by index
    if legal_actions and 0 <= action < len(legal_actions):
        return legal_actions[action]
    
    # Last resort: return as string
    return str(action)


def _convert_uno_step(obs, action, history):
    """Convert a single UNO step to SFT format."""
    try:
        # Calculate step from history length or use a counter
        step = len(obs.get('played_cards', [])) // 2
        
        # Get legal actions for potential fallback
        legal_actions = obs.get('legal_actions', [])
        
        # Convert action ID to string if needed
        action_str = convert_action_id_to_str(action, 'uno', legal_actions)
        
        item = prompt_uno % (
            obs.get('step', step),
            obs.get('current_player', 0),
            json.dumps(obs.get('hand', [])),
            json.dumps(obs.get('target', '')),
            json.dumps(obs.get('played_cards', [])),
            json.dumps(obs.get('num_cards', [])),
            json.dumps(history),
            json.dumps(legal_actions),
        )
        return {
            'instruction': item,
            'output': json.dumps({'action': action_str})
        }
    except Exception as e:
        return None


def _convert_gin_rummy_step(obs, action, history, step_idx=0):
    """Convert a single Gin Rummy step to SFT format.
    
    Handles the new format from gen_data.py where obs is a dict with:
    - hand: list of card strings like ["AS", "2H", "KD"]
    - top_discard: single card string or None
    - dead_cards: list of card strings
    - opponent_known_cards: list of card strings
    - stock_pile_num: int
    - legal_actions: list of action strings
    - player_id: int
    """
    try:
        # Get legal_actions
        legal_actions = obs.get('legal_actions', [])
        
        # Action from gen_data.py is already a string (e.g., "draw_card", "discard AS")
        if isinstance(action, str):
            action_str = action
        else:
            action_str = convert_action_id_to_str(action, 'gin_rummy', legal_actions)
        
        # Calculate step - use step_idx or history length
        step = step_idx if step_idx > 0 else len(history) if history else 0
        
        # Extract fields with defaults
        hand = obs.get('hand', [])
        top_discard = obs.get('top_discard', None)
        dead_cards = obs.get('dead_cards', [])
        opponent_known_cards = obs.get('opponent_known_cards', [])
        stock_pile_num = obs.get('stock_pile_num', 0)
        player_id = obs.get('player_id', 0)
        
        # Build prompt
        item = prompt_gin_rummy % (
            step,
            player_id,
            json.dumps(hand),
            json.dumps(top_discard),
            json.dumps(dead_cards),
            json.dumps(opponent_known_cards),
            json.dumps(stock_pile_num),
            json.dumps(history),
            json.dumps(legal_actions),
        )
        return {
            'instruction': item,
            'output': json.dumps({'action': action_str})
        }
    except Exception as e:
        print(f"Error converting gin rummy step: {e}, obs keys: {obs.keys() if isinstance(obs, dict) else 'not a dict'}")
        return None


def _convert_leduc_holdem_step(obs, action, history):
    """Convert a single Leduc Hold'em step to SFT format."""
    try:
        legal_actions = obs.get('legal_actions', [])
        action_str = convert_action_id_to_str(action, 'leduc_holdem', legal_actions)
        
        item = prompt_leduc_holdem % (
            obs.get('current_round', 0),
            obs.get('current_player', 0),
            json.dumps(obs.get('hand', '')),
            json.dumps(obs.get('public_card', '')),
            json.dumps(obs.get('my_chips', 0)),
            json.dumps(obs.get('all_chips', [])),
            json.dumps(obs.get('raise_nums', [])),
            json.dumps(history),
            json.dumps(legal_actions),
        )
        return {
            'instruction': item,
            'output': json.dumps({'action': action_str})
        }
    except Exception as e:
        return None


def _convert_limit_holdem_step(obs, action, history):
    """Convert a single Limit Hold'em step to SFT format."""
    try:
        legal_actions = obs.get('legal_actions', [])
        action_str = convert_action_id_to_str(action, 'limit_holdem', legal_actions)
        
        item = prompt_limit_holdem % (
            obs.get('current_round', 0),
            obs.get('current_player', 0),
            json.dumps(obs.get('hand', [])),
            json.dumps(obs.get('public_cards', [])),
            json.dumps(obs.get('my_chips', 0)),
            json.dumps(obs.get('all_chips', [])),
            json.dumps(obs.get('raise_nums', [])),
            json.dumps(history),
            json.dumps(legal_actions),
        )
        return {
            'instruction': item,
            'output': json.dumps({'action': action_str})
        }
    except Exception as e:
        return None


def _convert_nolimit_holdem_step(obs, action, history):
    """Convert a single No-Limit Hold'em step to SFT format."""
    try:
        legal_actions = obs.get('legal_actions', [])
        action_str = convert_action_id_to_str(action, 'nolimit_holdem', legal_actions)
        
        item = prompt_nolimit_holdem % (
            obs.get('stage', 'PREFLOP'),
            obs.get('current_player', 0),
            json.dumps(obs.get('hand', [])),
            json.dumps(obs.get('public_cards', [])),
            json.dumps(obs.get('my_chips', 0)),
            json.dumps(obs.get('all_chips', [])),
            json.dumps(obs.get('pot', 0)),
            json.dumps(obs.get('stakes', [])),
            json.dumps(history),
            json.dumps(legal_actions),
        )
        return {
            'instruction': item,
            'output': json.dumps({'action': action_str})
        }
    except Exception as e:
        return None


# ===========================================
# Other Game Converters (original format - unchanged)
# ===========================================

def split_by_game_reward(all_data):
    per_game = []
    cur_game = []
    for line in all_data:
        cur_game.append(line)
        if 'reward' in line:
            per_game.append(cur_game)
            cur_game = []
    return per_game

def convert_guandan(data):
    if not PROMPTS_AVAILABLE:
        return []
    all_data = read_jsonl(data)
    per_games = split_by_game_reward(all_data)

    items = []
    for game in per_games:
        if len(game) < 2:
            continue
        if game[-1]['reward'] > 0:
            for line in game[:-1]:
                if 'obs' not in line:
                    continue
                if len(line['obs']['legal_actions']) < 2:
                    continue
                item = prompt_guandan % (
                    json.dumps(line['obs']['my_pos']), 
                    json.dumps(line['obs']['my_hands']), 
                    json.dumps(line['obs']['remaining_hands']), 
                    json.dumps(line['obs']['last_action']), 
                    json.dumps(line['obs']['last_teammate_action']), 
                    json.dumps(line['obs']['number_of_cards_left']),
                    json.dumps(line['obs']['down_played_cards']),
                    json.dumps(line['obs']['teammate_played_cards']),
                    json.dumps(line['obs']['up_played_cards']),
                    json.dumps(line['obs']['self_rank']),
                    json.dumps(line['obs']['oppo_rank']),
                    json.dumps(line['obs']['cur_rank']),
                    json.dumps(line['obs']['legal_actions']),
                )
                sft_item = {
                    'instruction': item,
                    'output': json.dumps({'action': line['action']})
                }
                json_item = json.dumps(sft_item, ensure_ascii=False)
                items.append(json_item)
    return items

def convert_riichi(data):
    if not PROMPTS_AVAILABLE:
        return []
    all_data = read_jsonl(data)
    items = []
    for line in all_data:
        if line['obs']['rank'] > 0:
            continue
        if len(line['obs']['legal_actions']) < 2:
            continue
        item = prompt_riichi % (
            json.dumps(line['obs']['player_id']),
            json.dumps(line['obs']['bakaze']),
            json.dumps(line['obs']['jikaze']),
            json.dumps(line['obs']['kyoku']),
            json.dumps(line['obs']['honba']),
            json.dumps(line['obs']['kyotaku']),
            json.dumps(line['obs']['oya']),
            json.dumps(line['obs']['p_scores']), 
            json.dumps(line['obs']['p_rank']), 
            json.dumps(line['obs']['at_turn']), 
            json.dumps(line['obs']['tiles_left']),
            json.dumps(line['obs']['shanten']), 
            json.dumps(line['obs']['my_hands']),
            json.dumps(line['obs']['waits']), 
            json.dumps(line['obs']['dora_indicators']), 
            json.dumps(line['obs']['doras_owned']), 
            json.dumps(line['obs']['akas_in_hand']), 
            json.dumps(line['obs']['doras_seen']), 
            json.dumps(line['obs']['akas_seen']), 
            json.dumps(line['obs']['tiles_seen']), 
            json.dumps(line['obs']['ankan_candidates']), 
            json.dumps(line['obs']['kakan_candidates']), 
            json.dumps(line['obs']['kawa_overview']), 
            json.dumps(line['obs']['fuuro_overview']), 
            json.dumps(line['obs']['ankan_overview']), 
            json.dumps(line['obs']['last_tedashis']), 
            json.dumps(line['obs']['riichi_sutehais']), 
            json.dumps(line['obs']['last_drew_tile_self']), 
            json.dumps(line['obs']['last_discarded_tile']), 
            json.dumps(line['obs']['riichi_declared']), 
            json.dumps(line['obs']['riichi_accepted']), 
            json.dumps(line['obs']['can_w_riichi']), 
            json.dumps(line['obs']['is_w_riichi']), 
            json.dumps(line['obs']['at_furiten']), 
            json.dumps(line['obs']['is_menzen']), 
            json.dumps(line['obs']['legal_actions']),
        )
        sft_item = {
            'instruction': item,
            'output': json.dumps({'action': line['action']})
        }
        json_item = json.dumps(sft_item, ensure_ascii=False)
        items.append(json_item)
    return items

def convert_leduc_holdem(data):
    """Original format converter for leduc holdem."""
    if not PROMPTS_AVAILABLE:
        return []
    
    # Check if it's RLCard format
    if is_rlcard_data_file(data) or detect_file_format(data) == 'rlcard_txt':
        return convert_rlcard_format(data, 'leduc_holdem')
    
    items = []
    for line in read_jsonl_generator(data):
        if line and line[-1][-3] > 0:
            for sub_line in line:
                observation = sub_line[0]
                if len(observation['legal_actions']) < 2:
                    continue
                item = prompt_leduc_holdem % (
                    observation['current_round'],
                    observation['current_player'],
                    json.dumps(observation['hand']),
                    json.dumps(observation['public_card']),
                    json.dumps(observation['my_chips']),
                    json.dumps(observation['all_chips']),
                    json.dumps(observation['raise_nums']),
                    json.dumps(sub_line[-1]),
                    json.dumps(observation['legal_actions']),
                )
                sft_item = {
                    'instruction': item,
                    'output': json.dumps({'action': sub_line[1]})
                }
                json_item = json.dumps(sft_item, ensure_ascii=False)
                items.append(json_item)
    return items

def convert_limit_holdem(data):
    """Original format converter for limit holdem."""
    if not PROMPTS_AVAILABLE:
        return []
    
    # Check if it's RLCard format
    if is_rlcard_data_file(data) or detect_file_format(data) == 'rlcard_txt':
        return convert_rlcard_format(data, 'limit_holdem')
    
    items = []
    for line in read_jsonl_generator(data):
        if line and line[-1][-3] > 0:
            for sub_line in line:
                observation = sub_line[0]
                if len(observation['legal_actions']) < 2:
                    continue
                item = prompt_limit_holdem % (
                    observation['current_round'],
                    observation['current_player'],
                    json.dumps(observation['hand']),
                    json.dumps(observation['public_cards']),
                    json.dumps(observation['my_chips']),
                    json.dumps(observation['all_chips']),
                    json.dumps(observation['raise_nums']),
                    json.dumps(sub_line[-1]),
                    json.dumps(observation['legal_actions']),
                )
                sft_item = {
                    'instruction': item,
                    'output': json.dumps({'action': sub_line[1]})
                }
                json_item = json.dumps(sft_item, ensure_ascii=False)
                items.append(json_item)
    return items

def convert_nolimit_holdem(data):
    """Original format converter for no-limit holdem."""
    if not PROMPTS_AVAILABLE:
        return []
    
    # Check if it's RLCard format
    if is_rlcard_data_file(data) or detect_file_format(data) == 'rlcard_txt':
        return convert_rlcard_format(data, 'nolimit_holdem')
    
    items = []
    for line in read_jsonl_generator(data):
        if line and line[-1][-3] > 0:
            for sub_line in line:
                observation = sub_line[0]
                if len(observation['legal_actions']) < 2:
                    continue
                item = prompt_nolimit_holdem % (
                    observation['stage'],
                    observation['current_player'],
                    json.dumps(observation['hand']),
                    json.dumps(observation['public_cards']),
                    json.dumps(observation['my_chips']),
                    json.dumps(observation['all_chips']),
                    json.dumps(observation['pot']),
                    json.dumps(observation['stakes']),
                    json.dumps(sub_line[-1]),
                    json.dumps(observation['legal_actions']),
                )
                sft_item = {
                    'instruction': item,
                    'output': json.dumps({'action': sub_line[1]})
                }
                json_item = json.dumps(sft_item, ensure_ascii=False)
                items.append(json_item)
    return items

def convert_uno(data):
    """Convert UNO data - supports both original and RLCard formats."""
    if not PROMPTS_AVAILABLE:
        return []
    
    # Check if it's RLCard format
    if is_rlcard_data_file(data) or detect_file_format(data) == 'rlcard_txt':
        return convert_rlcard_format(data, 'uno')
    
    # Original format
    items = []
    for line in read_jsonl_generator(data):
        if line and line[-1][-3] > 0:
            for sub_line in line:
                observation = sub_line[0]
                if len(observation['legal_actions']) < 2:
                    continue
                item = prompt_uno % (
                    observation['step'],
                    observation['current_player'],
                    json.dumps(observation['hand']),
                    json.dumps(observation['target']),
                    json.dumps(observation['played_cards']),
                    json.dumps(observation['num_cards']),
                    json.dumps(sub_line[-1]),
                    json.dumps(observation['legal_actions']),
                )
                sft_item = {
                    'instruction': item,
                    'output': json.dumps({'action': sub_line[1]})
                }
                json_item = json.dumps(sft_item, ensure_ascii=False)
                items.append(json_item)
    return items

def convert_gin_rummy(data):
    """Convert Gin Rummy data - supports both original and RLCard formats."""
    if not PROMPTS_AVAILABLE:
        return []
    
    # Check if it's RLCard format
    if is_rlcard_data_file(data) or detect_file_format(data) == 'rlcard_txt':
        return convert_rlcard_format(data, 'gin_rummy')
    
    # Original format
    items = []
    for line in read_jsonl_generator(data):
        if line and line[-1][-3] > 0:
            for sub_line in line:
                observation = sub_line[0]
                item = prompt_gin_rummy % (
                    observation['step'],
                    observation['player_id'],
                    json.dumps(observation['hand']),
                    json.dumps(observation['top_discard']),
                    json.dumps(observation['dead_cards']),
                    json.dumps(observation['opponent_known_cards']),
                    json.dumps(observation['stock_pile_num']),
                    json.dumps(sub_line[-1]),
                    json.dumps(observation['legal_actions']),
                )
                sft_item = {
                    'instruction': item,
                    'output': json.dumps({'action': sub_line[1]})
                }
                json_item = json.dumps(sft_item, ensure_ascii=False)
                items.append(json_item)
    return items


# ===========================================
# Main Conversion Logic
# ===========================================

def convert_to_sft(data, game=None):
    """Convert a single file to SFT format."""
    if game is None:
        game = 'dou_dizhu'
    
    if game == 'dou_dizhu':
        return convert_dou_dizhu(data)
    elif game == 'leduc_holdem':
        return convert_leduc_holdem(data)
    elif game == 'limit_holdem':
        return convert_limit_holdem(data)
    elif game == 'nolimit_holdem':
        return convert_nolimit_holdem(data)
    elif game == 'uno':
        return convert_uno(data)
    elif game == 'guandan':
        return convert_guandan(data)
    elif game == 'riichi':
        return convert_riichi(data)
    elif game == 'gin_rummy':
        return convert_gin_rummy(data)
    else:
        print(f"Unknown game: {game}")
        return []

def write_sft(data, path):
    """Write SFT data to file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as sft_file:
        for item in data:
            sft_file.write(item)
            sft_file.write('\n')

def convert_dir(data_dir, game):
    """Convert all files in directory."""
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # Skip non-data files
            if file.endswith('.py') or file.endswith('.sh'):
                continue
            
            filepath = os.path.join(root, file)
            
            # Include trajectory files and log files (original formats)
            if file.startswith('trajectory') or file.startswith('log'):
                all_files.append((file, filepath))
            # NEW: Include RLCard format files (timestamp-worker-env_*.txt)
            elif is_rlcard_data_file(filepath):
                all_files.append((file, filepath))
            # Also check by content for .txt files
            elif file.endswith('.txt'):
                fmt = detect_file_format(filepath)
                if fmt in ['rlcard_txt', 'log_txt']:
                    all_files.append((file, filepath))

    # Sort files by file name
    all_files.sort(key=lambda x: x[0])
    print(f"Found {len(all_files)} data files")
    
    if not all_files:
        print(f"No data files found in {data_dir}")
        return []

    all_data = []
    
    # Multi-process with progress bar
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(convert_to_sft, file[1], game) for file in all_files]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                all_data.extend(result)
            except Exception as e:
                print(f"Warning: Error processing file: {e}")
                continue
    
    return all_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Convert data to SFT format")
    parser.add_argument(
        '--game',
        type=str,
        default='dou_dizhu',
        help='Game type: dou_dizhu, guandan, riichi, uno, leduc_holdem, limit_holdem, nolimit_holdem, gin_rummy'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='',
        help='Input file or directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='Output file path'
    )
    args = parser.parse_args()

    if not args.input:
        print("Error: --input is required")
        sys.exit(1)
    if not args.output:
        print("Error: --output is required")
        sys.exit(1)

    # Convert
    if os.path.isdir(args.input):
        data = convert_dir(args.input, args.game)
    elif os.path.isfile(args.input):
        data = convert_to_sft(args.input, args.game)
    else:
        raise ValueError(f"Input path '{args.input}' is neither a directory nor a file.")

    # Write output
    write_sft(data, args.output)
    
    print(f"\nConversion complete!")
    print(f"Total samples: {len(data)}")
    print(f"Output: {args.output}")