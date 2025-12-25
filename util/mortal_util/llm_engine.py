import json
from typing import *
from util.prompt_util import prompt_function_riichi, out_parse_function_riichi
from util.llm_client import llm_function
from util.mahjong_util.consts import convert_to_mjai_tile, obtain_legal_actions, convert_action_str_to_id
import random
from openai import OpenAI
from util.llm_config import model_config

def parse_riichi_state(event, mask):
    hand_cards = convert_to_mjai_tile(event.tehai, event.akas_in_hand)
    obs_data = {
        # 'obs': {
        #     'player_id': event.player_id,
        #     'oya': event.is_oya,
        #     'bakaze': event.bakaze,
        #     'kyoku': event.kyoku,
        #     'honba': event.honba,
        #     # 'my_pos': event.jikaze,
        #     'at_turn': event.at_turn,
        #     'my_hands': hand_cards,
        #     'last_drew_tile_self': event.last_self_tsumo(),
        #     'last_discarded_tile': event.last_kawa_tile(),
        #     'akas_in_hand': event.akas_in_hand,
        #     'shanten': event.shanten,
        #     'legal_actions': obtain_legal_actions(mask),
        # }
        'obs': {
            # information of the game
            'player_id': event.player_id,
            'bakaze': event.bakaze(),
            'jikaze': event.jikaze(),
            # Counts from 0 unlike mjai.
            'kyoku': event.kyoku,
            'honba': event.honba,
            'kyotaku': event.kyotaku,
            'oya': event.is_oya,
            # Rotated to be relative
            'p_scores': event.scores, # v4
            'p_rank': event.rank,
            'at_turn': event.at_turn,
            'tiles_left': event.tiles_left,
            'shanten': event.shanten,
            # information of own tiles
            'my_hands': hand_cards, # v34
            'waits': event.waits(), # v34
            # 'dora_factor': event.dora_factor, # v34
            'dora_indicators': event.dora_indicators(), # m5
            'doras_owned': event.doras_owned, # v4
            'akas_in_hand': event.akas_in_hand, # v3
            'doras_seen': event.doras_seen,
            'akas_seen': event.akas_seen, # v3
            'tiles_seen': event.tiles_seen(), # v34
            'keep_shanten_discards': event.keep_shanten_discards(), # v34
            'next_shanten_discards': event.next_shanten_discards(), # v34
            # 'forbidden_tiles': event.forbidden_tiles, # v34
            # 'discarded_tiles': event.discarded_tiles, # v34
            'ankan_candidates': event.ankan_candidates(),
            'kakan_candidates': event.kakan_candidates(),
            # information of known tiles
            'kawa': event.kawa(), # v4*m24
            'kawa_overview': event.kawa_overview(), # v4*m24
            'fuuro_overview': event.fuuro_overview(), # v4*m4*m4
            'ankan_overview': event.ankan_overview(), # v4*m4
            'last_tedashis': event.last_tedashis(), # v4
            'riichi_sutehais': event.riichi_sutehais(), # v4
            'last_drew_tile_self': event.last_self_tsumo(),
            'last_discarded_tile': event.last_kawa_tile(),
            # status information
            'riichi_declared': event.riichi_declared, # v4
            'riichi_accepted': event.riichi_accepted, # v4
            'can_w_riichi': event.can_w_riichi,
            'is_w_riichi': event.is_w_riichi,
            'at_furiten': event.at_furiten,
            'is_menzen': event.is_menzen,
            # 'has_next_shanten_discard': event.has_next_shanten_discard,
            'legal_actions': obtain_legal_actions(mask),
            # 'rank': int(player_ranks[at_kyoku[eid]+1]),
            # 'scores': list(scores_seq[at_kyoku[eid]+1]),
        }
    }
    return obs_data

class LLMEngine:
    def __init__(
        self,
        version,
        name = 'NoName',
        model = 'llm'
    ):
        self.engine_type = 'mortal2'
        self.version = version
        self.name = name

        self.model_config = model_config[model]
        self.client = OpenAI(
            base_url=self.model_config['call_func']['base_url'],
            api_key=self.model_config['call_func']['api_key'],
        )
        self.request_count = 0
        self.correct_count = 0

    def react_batch(self, obs, masks, obs_raw):
        return [self.step(obs_raw[i], masks[i]) for i in range(len(obs))]

    def step(self, state, mask):
        # print('test2')
        parsed = parse_riichi_state(state, mask)
        if len(parsed['obs']['legal_actions']) == 1:
            return convert_action_str_to_id(parsed['obs']['legal_actions'][0])

        prompt = prompt_function_riichi(parsed)
        output = llm_function(prompt, client=self.client, model_config=self.model_config)
        action = out_parse_function_riichi(output)

        self.request_count += 1

        if not action or action not in parsed['obs']['legal_actions']:
            # action = parsed['obs']['legal_actions'][0]
            action = random.choice(parsed['obs']['legal_actions'])
        else:
            self.correct_count += 1
        response = convert_action_str_to_id(action)
        return response
