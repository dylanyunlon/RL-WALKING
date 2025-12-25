import sys
import re
import os
import pandas as pd

# parse dou metric file
def parse_dou_metric(file_path):
    wp_results_pattern = re.compile(r'WP results:\s*landlord : Farmers - ([\d\.\-]+) : ([\d\.\-]+)')
    format_accuracy_pattern = re.compile(r'Format Accuracy results:(.*?)Time in minutes:', re.DOTALL)
    method_accuracy_pattern = re.compile(r'(\d+)\s+([\w-]+)\s+([\d\.\-]+)')
    time_pattern = re.compile(r'Time in minutes: (\d+\.\d+)')

    wp_results = None
    format_accuracy_results = []
    time_str = None

    with open(file_path, 'r') as file:
        contents = file.read()

        # Matching WP results
        wp_match = wp_results_pattern.search(contents)
        if wp_match:
            wp_results = (float(wp_match.group(1)), float(wp_match.group(2)))

        # Matching Format Accuracy results
        fa_match = format_accuracy_pattern.search(contents)
        if fa_match:
            fa_contents = fa_match.group(1)
            method_accuracy_matches = method_accuracy_pattern.findall(fa_contents)
            for match in method_accuracy_matches:
                method_name = match[1]
                accuracy = float(match[2])
                format_accuracy_results.append((method_name, accuracy))

        # Matching Time
        time_match = time_pattern.search(contents)
        if time_match:
            time_str = float(time_match.group(1))

    if wp_results is None or len(format_accuracy_results) != 3:
        return {}
    metric = {
        'landlord': wp_results[0],
        'farmers': wp_results[1],
        'model_0': format_accuracy_results[0][0],
        'model_1': format_accuracy_results[1][0],
        'model_2': format_accuracy_results[2][0],
        'format_accuracy_0': format_accuracy_results[0][1],
        'format_accuracy_1': format_accuracy_results[1][1],
        'format_accuracy_2': format_accuracy_results[2][1],
        'time': time_str
    }
    return metric

def parse_guandan_metric(file_path):
    win_rate_pattern = re.compile(r'\n(\d) ([\w-]+) Win rate: ([\d\.]+)')
    round_win_rate_pattern = re.compile(r'\n(\d) ([\w-]+) Round win rate: ([\d\.]+)')
    format_accuracy_pattern = re.compile(r'\n(\d) ([\w-]+) Format acc: ([\d\.]+)')
    time_pattern = re.compile(r'Time in minutes: (\d+\.\d+)')

    win_rate_results = None
    round_win_rate_results = None
    format_accuracy_results = None
    time_str = None

    with open(file_path, 'r') as file:
        contents = file.read()

        # Matching WP results
        match_result = win_rate_pattern.search(contents)
        if match_result:
            win_rate_results = (float(match_result.group(1)), match_result.group(2), float(match_result.group(3)))

        match_result = round_win_rate_pattern.search(contents)
        if match_result:
            round_win_rate_results = (float(match_result.group(1)), match_result.group(2), float(match_result.group(3)))

        # Matching Format Accuracy results
        match_result = format_accuracy_pattern.search(contents)
        if match_result:
            format_accuracy_results = (float(match_result.group(1)), match_result.group(2), float(match_result.group(3)))
        # Matching Time
        time_match = time_pattern.search(contents)
        if time_match:
            time_str = float(time_match.group(1))

    if win_rate_results is None:
        return {}
    metric = {
        'model': win_rate_results[1],
        'win_rate': win_rate_results[2],
        'round_win_rate': round_win_rate_results[2],
        'format_accuracy': format_accuracy_results[2],
        'time': time_str
    }
    return metric

def parse_riichi_metric(file_path):
    wp_results_pattern = re.compile(r'challenger rankings:.*?([\d\.\-]+), ([\d\.\-]+), ([\d\.\-]+)pt')
    format_accuracy_pattern = re.compile(r'Format Accuracy results:(.*?)Time in minutes:', re.DOTALL)
    method_accuracy_pattern = re.compile(r'(\d+)\s+([\w-]+)\s+([\d\.\-]+)')
    time_pattern = re.compile(r'Time in minutes: (\d+\.\d+)')

    wp_results = None
    format_accuracy_results = []
    time_str = None

    with open(file_path, 'r') as file:
        contents = file.read()

        # Matching WP results
        wp_match = wp_results_pattern.search(contents)
        if wp_match:
            wp_results = (float(wp_match.group(1)), float(wp_match.group(2)), float(wp_match.group(3)))

        # Matching Format Accuracy results
        fa_match = format_accuracy_pattern.search(contents)
        if fa_match:
            fa_contents = fa_match.group(1)
            method_accuracy_matches = method_accuracy_pattern.findall(fa_contents)
            for match in method_accuracy_matches:
                method_name = match[1]
                accuracy = float(match[2])
                format_accuracy_results.append((method_name, accuracy))
        # Matching Time
        time_match = time_pattern.search(contents)
        if time_match:
            time_str = float(time_match.group(1))

    if wp_results is None or len(format_accuracy_results) != 2:
        return {}
    metric = {
        'avg_win': wp_results[0],
        'avg_rank': wp_results[1],
        'avg_score': wp_results[2],
        'model_0': format_accuracy_results[0][0],
        'model_1': format_accuracy_results[1][0],
        'format_accuracy_0': format_accuracy_results[0][1],
        'format_accuracy_1': format_accuracy_results[1][1],
        'time': time_str
    }
    return metric

def parse_rlcard_metric(file_path):
    wp_results_pattern = re.compile(r'Rewards results:(.*?)Format Accuracy results:', re.DOTALL)
    format_accuracy_pattern = re.compile(r'Format Accuracy results:(.*?)Time in minutes:', re.DOTALL)
    method_result_pattern = re.compile(r'(\d+)\s+([\w-]+)\s+([\d\.\-]+)')
    time_pattern = re.compile(r'Time in minutes: (\d+\.\d+)')

    wp_results = []
    format_accuracy_results = []
    time_str = None

    with open(file_path, 'r') as file:
        contents = file.read()

        # Matching WP results
        wp_match = wp_results_pattern.search(contents)
        if wp_match:
            wp_contents = wp_match.group(1)
            wp_matches = method_result_pattern.findall(wp_contents)
            for match in wp_matches:
                method_name = match[1]
                reward = float(match[2])
                wp_results.append((method_name, reward))


        # Matching Format Accuracy results
        fa_match = format_accuracy_pattern.search(contents)
        if fa_match:
            fa_contents = fa_match.group(1)
            method_accuracy_matches = method_result_pattern.findall(fa_contents)
            for match in method_accuracy_matches:
                method_name = match[1]
                accuracy = float(match[2])
                format_accuracy_results.append((method_name, accuracy))
        # Matching Time
        time_match = time_pattern.search(contents)
        if time_match:
            time_str = float(time_match.group(1))

    metric = {}
    if wp_results is None or format_accuracy_results is None:
        return metric
    for i, (method_name, reward) in enumerate(wp_results):
        metric[f'model_{i}'] = method_name
        metric[f'reward_{i}'] = reward
        metric[f'format_accuracy_{i}'] = format_accuracy_results[i][1]
    metric['time'] = time_str
    return metric

# custom game order
game_order = ['doudizhu4', 'guandan4', 'riichi4', 'uno4', 'gin4','leduc4', 'limit4', 'nolimit4']

# extract game from file name
def extract_game(s):
    parts = s.split('-')
    if len(parts) >= 6:
        return parts[5]
    else:
        return None

# definition of sort key function, sort by game_order index
def sort_key(s):
    prefix = '-'.join(s['file'].split('-')[:5])
    game = extract_game(s['file'])
    game_index = game_order.index(game) if game in game_order else len(game_order)
    return (prefix, game_index)

def parse_metric_from_dir(input_dir, parse_func=parse_dou_metric, file_name_starts=None):
    all_result = []
    
    # Collect all file paths
    all_files = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isfile(item_path):
            if item.startswith(file_name_starts):
                all_files.append((item, item_path))

    # Sort files by file name
    all_files.sort(key=lambda x: x[0])
    if all_files == []:
        print('No file found')
        return

    for file, input_path in all_files:
        print(input_path)
        file_result = parse_func(input_path)
        file_result['file'] = file
        file_result['file_path'] = input_dir.split('/')[-1]
        all_result.append(file_result)

    all_result = sorted(all_result, key=sort_key)

    # save all result in csv format
    df = pd.DataFrame(all_result)
    df.to_csv(os.path.join(input_dir, f"metric-{input_dir.split('/')[-1]}-{all_files[-1][0]}.csv"), index=False)

if __name__ == '__main__':
    # input_file = '../logs_eval_ckpt-Qwen25-0_5B-Instruct'
    input_file = sys.argv[1]
    if len(sys.argv) > 2:
        mode = sys.argv[2]
    else:
        mode = 'std'
    if mode == 'fast':
        # fast result
        # douzero
        parse_metric_from_dir(input_file, parse_dou_metric, file_name_starts='doudizhu-44-100')
        # guandan
        parse_metric_from_dir(input_file, parse_guandan_metric, file_name_starts='guandan-44-20')
        parse_metric_from_dir(input_file, parse_guandan_metric, file_name_starts='guandan-44-10')
        # riichi
        parse_metric_from_dir(input_file, parse_riichi_metric, file_name_starts='riichi-44-20')
        # uno
        parse_metric_from_dir(input_file, parse_rlcard_metric, file_name_starts='uno-44-100')
        # limit
        parse_metric_from_dir(input_file, parse_rlcard_metric, file_name_starts='limit-44-100')
        # leduc
        parse_metric_from_dir(input_file, parse_rlcard_metric, file_name_starts='leduc-44-100')
        # nolimit
        parse_metric_from_dir(input_file, parse_rlcard_metric, file_name_starts='nolimit-44-100')
        # gin
        parse_metric_from_dir(input_file, parse_rlcard_metric, file_name_starts='gin-44-20')
    
    if mode == 'std':
        # standard result
        # douzero
        parse_metric_from_dir(input_file, parse_dou_metric, file_name_starts='doudizhu-44-500')
        # guandan
        parse_metric_from_dir(input_file, parse_guandan_metric, file_name_starts='guandan-44-40')
        parse_metric_from_dir(input_file, parse_guandan_metric, file_name_starts='guandan-44-20')
        # riichi
        parse_metric_from_dir(input_file, parse_riichi_metric, file_name_starts='riichi-44-100')
        parse_metric_from_dir(input_file, parse_riichi_metric, file_name_starts='riichi-44-50')
        # uno
        parse_metric_from_dir(input_file, parse_rlcard_metric, file_name_starts='uno-44-500')
        # limit
        parse_metric_from_dir(input_file, parse_rlcard_metric, file_name_starts='limit-44-1000')
        # leduc
        parse_metric_from_dir(input_file, parse_rlcard_metric, file_name_starts='leduc-44-1000')
        # nolimit
        parse_metric_from_dir(input_file, parse_rlcard_metric, file_name_starts='nolimit-44-1000')
        # gin
        parse_metric_from_dir(input_file, parse_rlcard_metric, file_name_starts='gin-44-100')