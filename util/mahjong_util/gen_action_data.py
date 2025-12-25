from libriichi.dataset import GameplayLoader
import os
from consts import convert_to_mjai_tile, convert_action_id_to_event, obtain_legal_actions
import json
import numpy as np
from tqdm import tqdm
import multiprocessing

def process_file(file_name, file_games, out_path):
    for game in file_games:
        # per move
        obs = game.take_obs()
        actions = game.take_actions()
        masks = game.take_masks()
        at_kyoku = game.take_at_kyoku()
        dones = game.take_dones()
        apply_gamma = game.take_apply_gamma()

        # per game
        grp = game.take_grp()
        player_id = game.take_player_id()

        game_size = len(obs)

        grp_feature = grp.take_feature()
        # rank_by_player = grp.take_rank_by_player()
        final_scores = grp.take_final_scores()
        scores_seq = np.concatenate((grp_feature[:, 3:] * 1e4, [final_scores]))
        rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
        player_ranks = rank_by_player_seq[:, player_id]

        for eid, event in enumerate(game.take_obs_raw()):
            hand_cards = convert_to_mjai_tile(event.tehai, event.akas_in_hand)
            # print(hand_cards)
            # print(masks[eid])
            # print(ALL_ACTIONS[actions[eid]])
            obs_action_data = {
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
                    'legal_actions': obtain_legal_actions(masks[eid]),
                    'rank': int(player_ranks[at_kyoku[eid]+1]),
                    # 'scores': list(scores_seq[at_kyoku[eid]+1]),
                },
                'action': convert_action_id_to_event(actions[eid]),
            }
            # print(obs_action_data['obs']['legal_actions'])
            with open(f'{out_path}/{file_name}-core.json', 'a', encoding="utf-8") as fout:
                fout.write(json.dumps(obs_action_data, ensure_ascii=False))
                fout.write('\n')
def process_file_parallel(args):
    file_name, file_games, out_path = args
    print(file_name)
    process_file(file_name, file_games, out_path)

def load_data(data_path, out_path):
    loader = GameplayLoader(
                version = 4,
                oracle = False,
                player_names = None,
                excludes = None,
                augmented = False,
            )
    
    # list files of the directory
    data_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    data_files_full = [os.path.join(data_path, f) for f in data_files]
    data = loader.load_gz_log_files(data_files_full)
    assert len(data) == len(data_files)

    for file_name, file_games in tqdm(zip(data_files, data)):
        print(file_name)
        process_file(file_name, file_games, out_path)

    # use multiprocessing Pool to process in parallel
    # inputs = [(file_name, file_games, out_path) for file_name, file_games in zip(data_files, data)]
    # with multiprocessing.Pool(processes=8) as pool:
    #     for _ in tqdm(pool.imap_unordered(process_file_parallel, inputs), total=len(inputs)):
    #         pass

if __name__ == '__main__':
    data_path = '/workspace/ww/project/card_agent/llm4cardgame/util/mahjong_util/logs'
    out_path = '/workspace/ww/project/card_agent/llm4cardgame/util/mahjong_util/logs_format4'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    load_data(data_path, out_path)