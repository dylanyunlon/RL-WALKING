import sqlite3
import bz2
import gzip
import json
from tqdm import tqdm

import xml.etree.ElementTree as ET
from mjlog2mjai import parse_mjlog_to_mjai, load_mjlog_from_str

# select one item from the database
def select_one(db_file, query):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(query)
    item = cursor.fetchall()
    conn.close()
    return item


if __name__ == '__main__':
    number = 7000
    db_file = './es4p.db'
    query = f'select log_id, log_content from logs limit {number}'
    item = select_one(db_file, query)

    for log_id, log_content in tqdm(item):
        mjai_data, dans = parse_mjlog_to_mjai(load_mjlog_from_str(bz2.decompress(log_content)))
        if mjai_data is None:
            print('mjai_data is None')
            continue
        # print(dans)
        mjai_str = '\n'.join(json.dumps(line, separators=(',', ':'), ensure_ascii=False) for line in mjai_data)
        # print(mjai_data)
        with gzip.open(f'./logs/{log_id}.json.gz', 'wb') as f:
            f.write(mjai_str.encode('utf-8'))
