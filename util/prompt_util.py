import json
import re
import numpy as np
from prompt.prompt_dou_dizhu4 import prompt_dou_dizhu
from prompt.prompt_guandan4 import prompt_guandan
from prompt.prompt_leduc_holdem import prompt_leduc_holdem
from prompt.prompt_limit_holdem import prompt_limit_holdem
from prompt.prompt_nolimit_holdem import prompt_nolimit_holdem
from prompt.prompt_mahjong_riichi4 import prompt_riichi
from prompt.prompt_uno import prompt_uno
from prompt.prompt_gin_rummy import prompt_gin_rummy

from rlcard.games.mahjong.card import MahjongCard
from rlcard.games.nolimitholdem.round import Action
from rlcard.games.nolimitholdem.game import Stage

class RLCardEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, MahjongCard):
            return obj.get_str()
        if isinstance(obj, Action):
            return obj.name
        if isinstance(obj, Stage):
            return obj.name
        # Handle numpy types
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def prompt_function_dou_dizhu(state):
    # item = prompt_dou_dizhu % (
    #         str(len(state.card_play_action_seq) // 3),
    #         state.player_position, 
    #         json.dumps(state.player_hand_cards), 
    #         json.dumps(state.played_cards)
    #     )
    # iterate each attributes of state
    item = prompt_dou_dizhu % (
            str(len(state.card_play_action_seq) // 3),
            state.player_position, 
            json.dumps(state.player_hand_cards), 
            json.dumps(state.other_hand_cards),
            json.dumps(state.last_move),
            json.dumps(state.played_cards),
            json.dumps(state.num_cards_left_dict),
            json.dumps(state.bomb_num),
            json.dumps(state.card_play_action_seq[-10:]),
            json.dumps(state.legal_actions),
        )
    return item

def out_parse_function(text):
    action = None
    try:
        json_objects = re.findall(r'\{.*?\}', text, re.DOTALL)
        parsed_objects = [json.loads(obj) for obj in json_objects]
        if parsed_objects:
            action = parsed_objects[-1]['action']
    except:
        print('error parsing action: ')
        print(text)
        pass
    return action

def out_parse_function_dou_dizhu(text):
    action = out_parse_function(text)
    if action is not None and isinstance(action, list):
        flag = [isinstance(a, int) for a in action]
        if action == [] or all(flag):
            action.sort()
        else: 
            action = None
    else:
        action = None
    return action

def prompt_function_guandan(state):
    observation = state['raw_obs']
    item = prompt_guandan % (
                    # json.dumps(line['obs']['turn_number']),
                    json.dumps(observation['my_pos']), 
                    json.dumps(observation['my_hands']), 
                    json.dumps(observation['remaining_hands']), 
                    json.dumps(observation['last_action']), 
                    json.dumps(observation['last_teammate_action']), 
                    json.dumps(observation['number_of_cards_left']),
                    json.dumps(observation['down_played_cards']),
                    json.dumps(observation['teammate_played_cards']),
                    json.dumps(observation['up_played_cards']),
                    json.dumps(observation['self_rank']),
                    json.dumps(observation['oppo_rank']),
                    json.dumps(observation['cur_rank']),
                    json.dumps(observation['legal_actions']),
                )
    return item

def out_parse_function_guandan(text):
    return out_parse_function(text)

def prompt_function_riichi(state):
    observation = state['obs']
    # item = prompt_riichi % (
    #         json.dumps(observation['player_id']),
    #         json.dumps(observation['oya']),
    #         json.dumps(observation['bakaze']),
    #         json.dumps(observation['kyoku']),
    #         json.dumps(observation['honba']),
    #         # json.dumps(line['obs']['my_pos']),
    #         json.dumps(observation['at_turn']), 
    #         json.dumps(observation['my_hands']), 
    #         json.dumps(observation['last_drew_tile_self']), 
    #         json.dumps(observation['last_discarded_tile']), 
    #         json.dumps(observation['akas_in_hand']),
    #         json.dumps(observation['shanten']),
    #         json.dumps(observation['legal_actions']),
    #     )
    item = prompt_riichi % (
            json.dumps(observation['player_id']),
            json.dumps(observation['bakaze']),
            json.dumps(observation['jikaze']),
            json.dumps(observation['kyoku']),
            json.dumps(observation['honba']),
            json.dumps(observation['kyotaku']),
            json.dumps(observation['oya']),
            json.dumps(observation['p_scores']), 
            json.dumps(observation['p_rank']), 
            json.dumps(observation['at_turn']), 
            json.dumps(observation['tiles_left']),
            json.dumps(observation['shanten']), 
            json.dumps(observation['my_hands']),
            json.dumps(observation['waits']), 
            json.dumps(observation['dora_indicators']), 
            json.dumps(observation['doras_owned']), 
            json.dumps(observation['akas_in_hand']), 
            json.dumps(observation['doras_seen']), 
            json.dumps(observation['akas_seen']), 
            json.dumps(observation['tiles_seen']), 
            json.dumps(observation['ankan_candidates']), 
            json.dumps(observation['kakan_candidates']), 
            json.dumps(observation['kawa_overview']), 
            json.dumps(observation['fuuro_overview']), 
            json.dumps(observation['ankan_overview']), 
            json.dumps(observation['last_tedashis']), 
            json.dumps(observation['riichi_sutehais']), 
            json.dumps(observation['last_drew_tile_self']), 
            json.dumps(observation['last_discarded_tile']), 
            json.dumps(observation['riichi_declared']), 
            json.dumps(observation['riichi_accepted']), 
            json.dumps(observation['can_w_riichi']), 
            json.dumps(observation['is_w_riichi']), 
            json.dumps(observation['at_furiten']), 
            json.dumps(observation['is_menzen']), 
            json.dumps(observation['legal_actions']),
        )
    return item

def out_parse_function_riichi(text):
    return out_parse_function(text)

def prompt_function_leduc_holdem(state):
    """
    Generate prompt for Leduc Hold'em.
    
    RLCard Leduc Holdem raw_obs 包含:
    - hand: 玩家手牌 (str, e.g., 'SJ', 'HQ', 'SK')
    - public_card: 公共牌 (str or None)
    - all_chips: 所有玩家的筹码 [player0_chips, player1_chips]
    - my_chips: 当前玩家的筹码
    - legal_actions: 合法动作列表 ['call', 'raise', 'fold', 'check']
    - current_player: 当前玩家ID
    
    需要推断的信息:
    - current_round: 从 public_card 推断 (None -> round 1, 有值 -> round 2)
    - raise_nums: 从 action_record 统计 raise 次数
    """
    observation = state['raw_obs']
    action_record = state.get('action_record', [])
    
    # 推断当前轮次: public_card 为 None 表示第一轮, 否则第二轮
    public_card = observation.get('public_card', None)
    current_round = 1 if public_card is None else 2
    
    # 统计 raise 次数 (从 action_record 中计算)
    # action_record 格式: [(player_id, action_str), ...]
    raise_nums = [0, 0]  # [round1_raises, round2_raises]
    current_round_for_counting = 1
    for record in action_record:
        if isinstance(record, tuple) and len(record) >= 2:
            _, action_str = record[0], record[1]
        else:
            action_str = str(record)
        
        # 检测是否进入第二轮 (当有 check 或 call 后下一个动作可能是新轮)
        # 简化处理: 通过当前 public_card 状态判断
        # 这里用一个简单的启发式: 如果已经在第二轮,之前的都算第一轮的
        if action_str == 'raise':
            if current_round == 1:
                raise_nums[0] += 1
            else:
                # 需要判断这个raise是哪一轮的
                # 简化: 假设前半部分是第一轮,后半部分是第二轮
                raise_nums[1] += 1
    
    # 更简单的方式: 直接统计总raise次数
    total_raises = sum(1 for r in action_record 
                       if (isinstance(r, tuple) and len(r) >= 2 and r[1] == 'raise') 
                       or (isinstance(r, str) and r == 'raise'))
    
    # 分配到两轮 (启发式: 如果在第二轮,假设一半一半,否则都在第一轮)
    if current_round == 1:
        raise_nums = [total_raises, 0]
    else:
        # 简单分配
        raise_nums = [min(total_raises, 2), max(0, total_raises - 2)]
    
    # 获取其他字段
    current_player = observation.get('current_player', 0)
    hand = observation.get('hand', '')
    my_chips = observation.get('my_chips', 0)
    all_chips = observation.get('all_chips', [0, 0])
    legal_actions = observation.get('legal_actions', [])
    
    item = prompt_leduc_holdem % (
                    current_round,
                    current_player,
                    json.dumps(hand),
                    json.dumps(public_card),
                    json.dumps(my_chips),
                    json.dumps(all_chips),
                    json.dumps(raise_nums),
                    json.dumps(action_record),
                    json.dumps(legal_actions),
                )
    return item

def out_parse_function_leduc_holdem(text):
    return out_parse_function(text)

def prompt_function_limit_holdem(state):
    """
    Generate prompt for Limit Hold'em.
    
    RLCard Limit Holdem raw_obs 可能包含:
    - hand: 玩家手牌
    - public_cards: 公共牌
    - all_chips: 所有玩家的筹码
    - my_chips: 当前玩家的筹码
    - legal_actions: 合法动作列表
    - current_player: 当前玩家ID
    
    需要检查并推断缺失的字段。
    """
    observation = state['raw_obs']
    action_record = state.get('action_record', [])
    
    # 尝试获取 current_round, 如果不存在则从 public_cards 推断
    if 'current_round' in observation:
        current_round = observation['current_round']
    else:
        # 根据 public_cards 数量推断轮次
        # Limit Hold'em: preflop(0), flop(3), turn(4), river(5)
        public_cards = observation.get('public_cards', [])
        if not public_cards:
            current_round = 0  # preflop
        elif len(public_cards) == 3:
            current_round = 1  # flop
        elif len(public_cards) == 4:
            current_round = 2  # turn
        else:
            current_round = 3  # river
    
    # 尝试获取 raise_nums, 如果不存在则计算
    if 'raise_nums' in observation:
        raise_nums = observation['raise_nums']
    else:
        # 从 action_record 统计
        raise_nums = sum(1 for r in action_record 
                        if (isinstance(r, tuple) and len(r) >= 2 and r[1] == 'raise') 
                        or (isinstance(r, str) and r == 'raise'))
    
    current_player = observation.get('current_player', 0)
    hand = observation.get('hand', [])
    public_cards = observation.get('public_cards', [])
    my_chips = observation.get('my_chips', 0)
    all_chips = observation.get('all_chips', [0, 0])
    legal_actions = observation.get('legal_actions', [])
    
    item = prompt_limit_holdem % (
                    current_round,
                    current_player,
                    json.dumps(hand),
                    json.dumps(public_cards),
                    json.dumps(my_chips),
                    json.dumps(all_chips),
                    json.dumps(raise_nums),
                    json.dumps(action_record),
                    json.dumps(legal_actions),
                )
    return item

def out_parse_function_limit_holdem(text):
    return out_parse_function(text)

def prompt_function_nolimit_holdem(state):
    observation = state['raw_obs']
    item = prompt_nolimit_holdem % (
                    observation['stage'].name,
                    int(observation['current_player']),
                    json.dumps(observation['hand'], cls=RLCardEncoder),
                    json.dumps(observation['public_cards'], cls=RLCardEncoder),
                    json.dumps(int(observation['my_chips'])),
                    json.dumps([int(x) for x in observation['all_chips']]),
                    json.dumps(int(observation['pot'])),         # ← 关键修复
                    json.dumps([int(x) for x in observation['stakes']]),
                    json.dumps(state['action_record'], cls=RLCardEncoder),
                    json.dumps(observation['legal_actions'], cls=RLCardEncoder),
                )
    return item

def out_parse_function_nolimit_holdem(text):
    return out_parse_function(text)

def prompt_function_uno(state):
    observation = state['raw_obs']
    item = prompt_uno % (
                    observation['current_player'],
                    json.dumps(observation['hand']),
                    json.dumps(observation['target']),
                    json.dumps(observation['played_cards']),
                    json.dumps(observation['num_cards']),
                    json.dumps(state['action_record']),
                    json.dumps(observation['legal_actions']),
                )
    return item

def out_parse_function_uno(text):
    return out_parse_function(text)

def prompt_function_gin_rummy(state):
    """
    Generate prompt for Gin Rummy.
    
    RLCard gin-rummy 的 state 结构:
    - state['obs']: numpy array 观测向量
    - state['raw_obs']: GinRummyObservation 对象 (不是字典!)
    - state['legal_actions']: OrderedDict
    - state['raw_legal_actions']: list of ActionEvent
    - state['action_record']: list of action history
    
    GinRummyObservation 对象有以下属性:
    - player_id: int
    - hand: list of Card
    - top_discard: Card or None
    - dead_cards: list of Card
    - opponent_known_cards: list of Card
    - stock_pile_num: int (牌堆剩余数量)
    - legal_actions: list of ActionEvent
    """
    # 获取 raw_obs (可能是对象或字典)
    raw_obs = state['raw_obs']
    action_record = state.get('action_record', [])
    
    # 计算当前步数 (从 action_record 推断)
    current_step = len(action_record)
    
    # 处理 GinRummyObservation 对象
    if hasattr(raw_obs, 'player_id'):
        # 是对象格式
        obs_dict = {
            'player_id': raw_obs.player_id,
            'hand': [str(card) for card in raw_obs.hand] if hasattr(raw_obs.hand, '__iter__') else [],
            'top_discard': str(raw_obs.top_discard) if raw_obs.top_discard else None,
            'dead_cards': [str(card) for card in raw_obs.dead_cards] if hasattr(raw_obs.dead_cards, '__iter__') else [],
            'opponent_known_cards': [str(card) for card in raw_obs.opponent_known_cards] if hasattr(raw_obs.opponent_known_cards, '__iter__') else [],
            'stock_pile_num': raw_obs.stock_pile_num if hasattr(raw_obs, 'stock_pile_num') else 0,
            'legal_actions': [str(action) for action in raw_obs.legal_actions] if hasattr(raw_obs.legal_actions, '__iter__') else []
        }
        
        player_id = obs_dict['player_id']
        hand = obs_dict['hand']
        top_discard = obs_dict['top_discard']
        dead_cards = obs_dict['dead_cards']
        opponent_known_cards = obs_dict['opponent_known_cards']
        stock_pile_num = obs_dict['stock_pile_num']
        legal_actions = obs_dict['legal_actions']
    else:
        # 如果是字典格式（不应该发生，但作为后备）
        player_id = raw_obs.get('player_id', raw_obs.get('current_player', 0))
        hand = raw_obs.get('hand', [])
        top_discard = raw_obs.get('top_discard', raw_obs.get('discard_pile', [''])[0] if raw_obs.get('discard_pile') else '')
        dead_cards = raw_obs.get('dead_cards', raw_obs.get('known_cards', []))
        opponent_known_cards = raw_obs.get('opponent_known_cards', [])
        stock_pile_num = raw_obs.get('stock_pile_num', raw_obs.get('stock_pile', 52))
        legal_actions = raw_obs.get('legal_actions', [])
    
    # 同时转换 action_record 中的动作为可读格式
    readable_action_record = []
    for action in action_record:
        if hasattr(action, '__str__') and 'ActionEvent' in type(action).__name__:
            readable_action_record.append(str(action))
        else:
            readable_action_record.append(str(action) if not isinstance(action, str) else action)
    
    item = prompt_gin_rummy % (
                    current_step,
                    player_id,
                    json.dumps(hand),
                    json.dumps(top_discard),
                    json.dumps(dead_cards),
                    json.dumps(opponent_known_cards),
                    json.dumps(stock_pile_num),
                    json.dumps(readable_action_record, cls=RLCardEncoder),
                    json.dumps(legal_actions),
                )
    return item

def out_parse_function_gin_rummy(text):
    return out_parse_function(text)

# str to specific action type
def str_to_dou_action(text):
    return text

def str_to_guandan_action(text):
    return text

def str_to_riichi_action(text):
    return text

def str_to_leduc_holdem_action(text):
    return text

def str_to_limit_holdem_action(text):
    return text

def str_to_nolimit_holdem_action(text):
    return text

def str_to_uno_action(text):    
    return text

def str_to_gin_rummy_action(text):
    """
    Convert string action or int action ID to Gin Rummy ActionEvent.
    
    Input can be:
    - int: action ID directly (from random fallback)
    - str: action string from LLM like "draw_card", "discard AS", etc.
    
    For RLCard, we need to return the ActionEvent object.
    
    ===== RLCard Gin Rummy 正确的 Action ID 映射 =====
    ID  0: score N (score_player_0)
    ID  1: score S (score_player_1)
    ID  2: draw_card
    ID  3: pick_up_discard
    ID  4: declare_dead_hand
    ID  5: gin
    ID  6-18: discard Spades (AS=6, 2S=7, ..., KS=18)
    ID 19-31: discard Hearts (AH=19, 2H=20, ..., KH=31)
    ID 32-44: discard Diamonds (AD=32, 2D=33, ..., KD=44)
    ID 45-56: discard Clubs (AC=45, 2C=46, ..., QC=56)
    
    Card ID 计算: card_id = suit * 13 + rank
        Spades (S): suit=0, card_id = 0-12
        Hearts (H): suit=1, card_id = 13-25
        Diamonds (D): suit=2, card_id = 26-38
        Clubs (C): suit=3, card_id = 39-51 (但只到 QC=50)
    
    discard action_id = 6 + card_id
    """
    from rlcard.games.gin_rummy.utils.action_event import ActionEvent
    
    # Handle integer input (from random fallback when LLM fails)
    if isinstance(text, int):
        try:
            return ActionEvent.decode_action(text)
        except Exception as e:
            print(f'Error decoding action ID {text}: {e}')
            return ActionEvent.decode_action(2)  # Fallback to draw_card (ID=2)
    
    # Handle None or empty
    if text is None:
        return ActionEvent.decode_action(2)  # draw_card (ID=2)
    
    # Handle string input
    if not isinstance(text, str):
        text = str(text)
    
    text = text.strip()
    
    # Card mapping for parsing card strings
    # Suits: S=0, H=1, D=2, C=3 (每个 suit 有 13 张牌)
    suits = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
    # Ranks: A=0, 2=1, 3=2, ..., T=9, J=10, Q=11, K=12
    ranks = {'A': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, 
             '8': 7, '9': 8, 'T': 9, '10': 9, 'J': 10, 'Q': 11, 'K': 12}
    
    def card_str_to_id(card_str):
        """
        Convert card string like 'AS' or '6S' to card ID (0-51).
        Card ID = suit * 13 + rank
        例如: AS = 0*13 + 0 = 0
              6S = 0*13 + 5 = 5
              AH = 1*13 + 0 = 13
              QC = 3*13 + 11 = 50
        """
        card_str = card_str.strip().upper()
        if len(card_str) < 2:
            return None
        
        # Handle 10 specially (written as "10S" or "TS")
        if card_str.startswith('10'):
            rank_str = '10'
            suit_str = card_str[2] if len(card_str) > 2 else ''
        else:
            # 通常格式: rank + suit, 如 "6S", "QC"
            # rank 是除了最后一个字符以外的部分
            rank_str = card_str[:-1]
            suit_str = card_str[-1]
        
        if suit_str not in suits or rank_str not in ranks:
            return None
        
        return suits[suit_str] * 13 + ranks[rank_str]
    
    # Try to parse common action patterns
    try:
        # ===== 基本动作 (正确的 ID) =====
        
        # draw_card action (action_id = 2)
        if text == 'draw_card':
            return ActionEvent.decode_action(2)
        
        # pick_up_discard action (action_id = 3)  
        if text == 'pick_up_discard':
            return ActionEvent.decode_action(3)
        
        # declare_dead_hand action (action_id = 4)
        # 注意: 训练数据用 "declare_dead", RLCard 用 "declare_dead_hand"
        if text in ['declare_dead', 'declare_dead_hand']:
            return ActionEvent.decode_action(4)
        
        # gin action (action_id = 5)
        if text == 'gin':
            return ActionEvent.decode_action(5)
        
        # score actions (action_id = 0, 1)
        if text in ['score N', 'score_player_0', 'score player 0']:
            return ActionEvent.decode_action(0)
        if text in ['score S', 'score_player_1', 'score player 1']:
            return ActionEvent.decode_action(1)
        
        # ===== discard 动作 =====
        # discard actions: "discard AS" -> action_id = 6 + card_id
        if text.startswith('discard '):
            card_str = text[8:].strip()
            card_id = card_str_to_id(card_str)
            if card_id is not None:
                # Discard actions 从 ID 6 开始
                action_id = 6 + card_id
                return ActionEvent.decode_action(action_id)
        
        # ===== knock 动作 =====
        # knock 在 RLCard gin-rummy 中实际上等同于 discard + knock flag
        # 但在评估时，knock 通常不会出现在 legal_actions 中
        # 如果出现，我们按 discard 处理
        if text.startswith('knock '):
            card_str = text[6:].strip()
            card_id = card_str_to_id(card_str)
            if card_id is not None:
                action_id = 6 + card_id
                return ActionEvent.decode_action(action_id)
            
    except Exception as e:
        print(f'Error parsing gin rummy action "{text}": {e}')
    
    # Fallback: return draw_card action
    print(f'Warning: Could not parse gin rummy action "{text}", using draw_card')
    return ActionEvent.decode_action(2)  # draw_card (ID=2)

#     # Rewards results:
# 0 llm-lora -0.515
# 1 gin-rummy-novice-rule 0.180
# Format Accuracy results:
# 0 llm-lora 1.000
# 1 gin-rummy-novice-rule -1.000
# Time in minutes: 5.858792050679525
# [✓] Evaluation complete