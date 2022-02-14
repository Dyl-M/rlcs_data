# -*- coding: utf-8 -*-

import json
import data_collection_tools

"""File Information

@file_name: data_collection.py
@author: Dylan "dyl-m" Monfret

Data collection main script.
"""

"GLOBAL"

with open('../../data/private/my_token.txt', 'r', encoding='utf8') as token_file:
    my_token = token_file.read()

try:
    with open('../../data/retrieved/raw.json', 'r', encoding='utf8') as raw_dataset_file:
        raw = json.load(raw_dataset_file)
except FileNotFoundError:
    raw = None

RLCS_ALL = 'rlcs-2021-22-6d4xifwwqz'

"MAIN"

if __name__ == '__main__':
    # Step 1: iteration on `RLCS_ALL` group
    rlcs_replays = data_collection_tools.exploring_group(RLCS_ALL, my_token)
    with open('../../data/retrieved/pre_dataset.json', 'w', encoding='utf-8') as pre_dataset_file:
        json.dump(rlcs_replays, pre_dataset_file, indent=4)  # Saving intermediate results

    # Step 2: collect details from each replay
    with open('../../data/retrieved/pre_dataset.json', 'r', encoding='utf8') as pre_dataset_file:
        pre_dataset = json.load(pre_dataset_file)
    dataset = data_collection_tools.add_details(pre_dataset, raw, my_token)  # workers=8
    with open('../../data/retrieved/raw.json', 'w', encoding='utf-8') as dataset_file:
        json.dump(dataset, dataset_file, indent=4)  # Saving final results
