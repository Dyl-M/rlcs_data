# -*- coding: utf-8 -*-

import json
import requests

from time import sleep

"""File Informations

@file_name: retrieving_step2.py
@author: Dylan "dyl-m" Monfret
"""

"PREPARATORY ELEMENTS"

with open('../../data/private/my_token.txt', 'r', encoding='utf8') as token_file:
    my_token = token_file.read()

with open('../../data/retrieved/pre_dataset.json', 'r', encoding='utf8') as pre_dataset_file:
    pre_dataset = json.load(pre_dataset_file)

"FUNCTIONS"


def get_replay_stats(replay_id, token):
    """Get detailed replay's stats.

    :param replay_id: ballchasing.com replay ID.
    :param token: ballchasing.com API token.
    :return replay_stats:
    """
    replay_url = f'https://ballchasing.com/api/replays/{replay_id}'
    headers = {'Authorization': token}

    try:
        request = requests.get(replay_url, headers=headers)
        replay_stats = request.json()

        replay_stats.pop('id')
        replay_stats['link'] = replay_stats['link'].replace('api/replays', 'replay')

        for group in replay_stats['groups']:
            group['link'] = group['link'].replace('api/groups', 'group')

        return replay_stats

    except json.decoder.JSONDecodeError:
        print("Error 500: Resources unavailable")
        return None


def add_details(replay_list, token):
    """Add detailed stats to replays.

    :param replay_list: list of replays without details.
    :param token: ballchasing.com API token.
    :return replay_list: list of replays with details added into 'details' field.
    """
    for num, a_replay in enumerate(replay_list):
        print(f'REPLAY ID: {a_replay["ballchasing_id"]}')
        a_replay['details'] = get_replay_stats(a_replay['ballchasing_id'], token)

        if (num + 1 % 2) == 0:
            sleep(1)

    return replay_list


"MAIN"

if __name__ == '__main__':
    dataset = add_details(pre_dataset, my_token)
    with open('../../data/retrieved/raw.json', 'w', encoding='utf-8') as dataset_file:
        json.dump(dataset, dataset_file, indent=4)
