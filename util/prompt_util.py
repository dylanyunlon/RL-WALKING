import json
import re
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
    observation = state['raw_obs']
    item = prompt_leduc_holdem % (
                    observation['current_round'],
                    observation['current_player'],
                    json.dumps(observation['hand']),
                    json.dumps(observation['public_card']),
                    json.dumps(observation['my_chips']),
                    json.dumps(observation['all_chips']),
                    json.dumps(observation['raise_nums']),
                    json.dumps(state['action_record']),
                    json.dumps(observation['legal_actions']),
                )
    return item

def out_parse_function_leduc_holdem(text):
    return out_parse_function(text)

def prompt_function_limit_holdem(state):
    observation = state['raw_obs']
    item = prompt_limit_holdem % (
                    observation['current_round'],
                    observation['current_player'],
                    json.dumps(observation['hand']),
                    json.dumps(observation['public_cards']),
                    json.dumps(observation['my_chips']),
                    json.dumps(observation['all_chips']),
                    json.dumps(observation['raise_nums']),
                    json.dumps(state['action_record']),
                    json.dumps(observation['legal_actions']),
                )
    return item

def out_parse_function_limit_holdem(text):
    return out_parse_function(text)

def prompt_function_nolimit_holdem(state):
    observation = state['raw_obs']
    item = prompt_nolimit_holdem % (
                    observation['stage'].name,
                    observation['current_player'],
                    json.dumps(observation['hand']),
                    json.dumps(observation['public_cards']),
                    json.dumps(observation['my_chips']),
                    json.dumps(observation['all_chips']),
                    json.dumps(observation['pot']),
                    json.dumps(observation['stakes']),
                    json.dumps(state['action_record'], cls=RLCardEncoder),
                    json.dumps(observation['legal_actions'], cls=RLCardEncoder),
                )
    return item

def out_parse_function_nolimit_holdem(text):
    return out_parse_function(text)

def prompt_function_uno(state):
    """
    为 UNO 游戏生成 prompt
    RLCard UNO 的 raw_obs 实际包含:
    - hand: 玩家手牌列表
    - target: 当前弃牌堆顶牌
    - played_cards: 已打出的牌
    - legal_actions: 合法动作列表
    - num_cards: 各玩家剩余牌数
    - num_players: 玩家总数
    - current_player: 当前玩家ID
    """
    observation = state['raw_obs']
    
    # 使用 action_record 长度作为步数估计（因为 RLCard 不提供 step 字段）
    current_step = len(state.get('action_record', []))
    
    # 从 raw_obs 获取实际存在的字段
    current_player = observation.get('current_player', 0)
    hand = observation.get('hand', [])
    target = observation.get('target', '')
    played_cards = observation.get('played_cards', [])
    num_cards = observation.get('num_cards', [])
    action_record = state.get('action_record', [])
    legal_actions = observation.get('legal_actions', [])
    
    item = prompt_uno % (
                    current_step,
                    current_player,
                    json.dumps(hand),
                    json.dumps(target),
                    json.dumps(played_cards),
                    json.dumps(num_cards),
                    json.dumps(action_record),
                    json.dumps(legal_actions),
                )
    return item

def out_parse_function_uno(text):
    return out_parse_function(text)

def prompt_function_gin_rummy(state):
    """
    为 Gin Rummy 游戏生成 prompt
    需要检查 RLCard Gin Rummy 环境的实际数据结构
    """
    observation = state['raw_obs']
    
    # 使用 action_record 长度作为步数估计
    current_step = len(state.get('action_record', []))
    
    # Gin Rummy 可能的字段（需要根据实际环境调整）
    player_id = observation.get('player_id', observation.get('current_player', 0))
    hand = observation.get('hand', [])
    
    # Gin Rummy 特有字段，提供默认值以防不存在
    top_discard = observation.get('top_discard', observation.get('discard_pile', [''])[0] if observation.get('discard_pile') else '')
    dead_cards = observation.get('dead_cards', observation.get('known_cards', []))
    opponent_known_cards = observation.get('opponent_known_cards', [])
    stock_pile_num = observation.get('stock_pile_num', observation.get('stock_pile', 52))
    
    action_record = state.get('action_record', [])
    legal_actions = observation.get('legal_actions', [])
    
    item = prompt_gin_rummy % (
                    current_step,
                    player_id,
                    json.dumps(hand),
                    json.dumps(top_discard),
                    json.dumps(dead_cards),
                    json.dumps(opponent_known_cards),
                    json.dumps(stock_pile_num),
                    json.dumps(action_record),
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
    return Action[text]

def str_to_uno_action(text):    
    return text

def str_to_gin_rummy_action(text):
    return text