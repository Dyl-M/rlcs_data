# -*- coding: utf-8 -*-

import json
import os
import requests

from datetime import datetime, timezone
from time import sleep

"""File Information

@file_name: api_calls.py
@author: Dylan "dyl-m" Monfret

Objective: #TODO

Summary: #TODO
"""

"- PREPARATORY ELEMENTS -"

with open('../data/private/my_token.txt', 'r', encoding='utf8') as token_file:
    my_token = token_file.read()

replay_url = 'https://ballchasing.com/api/replays/'

today = datetime.today()

""" Playlist choices:
- Ranked modes: ranked-duels / ranked-doubles / ranked-solo-standard / ranked-standard / tournament
- Unranked modes: unranked-duels / unranked-doubles / unranked-standard / unranked-chaos
- Extra modes: hoops / rumble / dropshot / snowday / dropshot-rumble / rocketlabs / heatseeker
- Ranked Extra modes: ranked-hoops / ranked-rumble / ranked-dropshot / ranked-snowday
- Offline modes: season / offline
- private
"""

# If you want to test this code using the API

with open('../data/public/seasons.json', 'r', encoding='utf8') as seasons_file:
    seasons = json.load(seasons_file)

seasons_reverse = seasons[::-1]

"- LOCAL FUNCTIONS -"


def add_to_list(api_call):
    """Change API request to a list of replays by managing potential KeyError.

    :param api_call: Request object
    :return: list of replays.
    """
    the_list = api_call.json()['list']

    return the_list


def get_replays(token, season_name, season_alt_name, season_code, ref_date, start_date, end_date):
    """Get replays from ballchasing.com with filters.

    :param token: API Key.
    :param season_name: Rocket Pass season name on ballchasing.com.
    :param season_alt_name: Rocket Pass season name on ballchasing.com, starting from Season 15 / Season 1 F2P.
    :param season_code: Code associated to season number (0 if before S1, None if after latest season).
    :param ref_date: Reference date to collect data (replays will be collected if uploaded before this date).
    :param start_date: Season's starting date.
    :param end_date: Season's endind date.
    :return replays_list: list of replays.
    """
    utc_tz = timezone.utc

    ref_date_code = ref_date.astimezone(utc_tz).strftime('%Y%m%d%H%M%S')
    ref_date_str = ref_date.astimezone(utc_tz).strftime('%Y-%m-%dT%H:%M:%S+%z')

    start_date_as_utc = datetime.strptime(start_date, '%Y-%m-%d').astimezone(utc_tz)
    end_date_as_utc = datetime.strptime(end_date, '%Y-%m-%d').astimezone(utc_tz)

    # TODO: print made to avoid "unused variable" warning.
    print(start_date_as_utc)
    print(end_date_as_utc)

    if season_alt_name is None:
        json_file_path = f'{ref_date_code}_{season_name}.json'
    else:
        json_file_path = f'{ref_date_code}_{season_name}_ALT_{season_alt_name}.json'

    if season_code == '0' or season_code is None:
        headers = {'Authorization': token, 'created-before': ref_date_str, 'count': '200'}
    else:
        headers = {'Authorization': token, 'season': season_code, 'created-before': ref_date_str, 'count': '200'}

    count = 1
    replays_list = []
    replay_request = requests.get(replay_url, headers=headers)
    replays_list += add_to_list(replay_request)
    next_url = replay_request.json()['next']

    print(f"Pages seen: {count:06d} | {replay_request} | Replays saved: {len(replays_list)}")

    while next_url:

        if count % 2 == 0:
            sleep(1)

        replay_request = requests.get(next_url, headers={'Authorization': token})

        if replay_request.status_code != 429:

            replays_list += add_to_list(replay_request)
            next_url = replay_request.json()['next']

        else:

            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(replays_list, json_file)

            file_size = os.stat(json_file_path).st_size / (1024 * 1024)

            print(f"Pages seen: {count + 1:06d} | "
                  f"{replay_request} | "
                  f"Replays saved: {len(replays_list)} ({file_size:.2f} MB)")

            if count <= 3600:
                sleep(count)
            else:
                sleep(3600)

            replay_request = requests.get(next_url, headers={'Authorization': token})
            replays_list += add_to_list(replay_request)
            next_url = replay_request.json()['next']

        print(f"Pages seen: {count + 1:06d} | {replay_request} | Replays saved: {len(replays_list)}")
        count += 1

    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(replays_list, json_file)

    return replays_list


def api_full_retrieving(token, season_list, ref_date):
    """Get all replays from ballchasing.com.

    :param token: API Key.
    :param season_list: list of competitive season in Rocket League.
    :param ref_date: Reference date to collect data (replays will be collected if uploaded before this date).
    """
    for a_season in season_list:
        get_replays(token=token,
                    season_name=a_season['season_name'],
                    season_alt_name=a_season['season_alt_name'],
                    season_code=a_season['season_code'],
                    ref_date=ref_date,
                    start_date=a_season['start_date'],
                    end_date=a_season['end_date'])


"- MAIN -"

if __name__ == '__main__':
    tz = timezone.utc
    print(seasons_reverse[1])
    get_replays(my_token,
                seasons_reverse[1]['season_name'],
                seasons_reverse[1]['season_alt_name'],
                seasons_reverse[1]['season_code'],
                today,
                seasons_reverse[1]['start_date'],
                seasons_reverse[1]['end_date'])
