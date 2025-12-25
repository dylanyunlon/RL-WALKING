MJAI_VEC34_TILES = [
    "1m",
    "2m",
    "3m",
    "4m",
    "5m",
    "6m",
    "7m",
    "8m",
    "9m",
    "1p",
    "2p",
    "3p",
    "4p",
    "5p",
    "6p",
    "7p",
    "8p",
    "9p",
    "1s",
    "2s",
    "3s",
    "4s",
    "5s",
    "6s",
    "7s",
    "8s",
    "9s",
    "E",
    "S",
    "W",
    "N",
    "P",
    "F",
    "C",
    "5mr",
    "5pr",
    "5sr",
]

ID_TO_ACTIONS = [
    'reach',
    'chi_low',
    'chi_mid',
    'chi_high',
    'pon',
    'kan',
    'hora',
    'ryukyoku',
    'pass',
]

ALL_ACTIONS = MJAI_VEC34_TILES + ID_TO_ACTIONS
ALL_ACTIONS_STR_TO_ID = {v: k for k,v in enumerate(ALL_ACTIONS)}


def convert_to_mjai_tile(tehai, akas_in_hand):
    tiles = []
    for tile_idx, tile_count in enumerate(tehai):
        if tile_idx == 4 and akas_in_hand[0]:
            tile_count -= 1
            tiles.append("5mr")
        elif tile_idx == 4 + 9 and akas_in_hand[1]:
            tile_count -= 1
            tiles.append("5pr")
        elif tile_idx == 4 + 18 and akas_in_hand[2]:
            tile_count -= 1
            tiles.append("5sr")

        for _ in range(tile_count):
            tiles.append(MJAI_VEC34_TILES[tile_idx])
    return tiles

def obtain_legal_actions(mask):
    legal_actions = []
    for action_id, is_legal in enumerate(mask):
        if is_legal:
            if action_id < 37:
                legal_actions.append(f'dahai:{ALL_ACTIONS[action_id]}')
            else:
                legal_actions.append(ALL_ACTIONS[action_id])
    return legal_actions

def convert_action_id_to_event(action_id):
    event_string = ALL_ACTIONS[action_id]
    if action_id < 37:
        event_string = 'dahai:' + event_string
    return event_string

def convert_action_str_to_id(action_str):
    if action_str.startswith('dahai:'):
        action_str = action_str.split('dahai:')[-1]
    if action_str in ALL_ACTIONS_STR_TO_ID:
        action_id = ALL_ACTIONS_STR_TO_ID[action_str]
    else:
        action_id = ALL_ACTIONS_STR_TO_ID['pass']
    return action_id