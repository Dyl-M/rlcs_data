# -*- coding: utf-8 -*-

import json
import requests

from p_tqdm import p_map

# from time import sleep | Unnecessary with Patreon Tier 3+ support ( ^_^)

"""File Information

@file_name: data_collection_tools.py
@author: Dylan "dyl-m" Monfret

Data collection functions.
"""

"FUNCTIONS"


# TODO: Add a process for "Closed Qualifiers".

def get_groups(group_id: str, token: str):
    """Get children groups and direct replays summaries from a ballchasing.com parent groups
    :param group_id: ballchasing.com group ID
    :param token: ballchasing.com API token
    :return: children groups and direct replays summaries
    """
    parent_group_uri = f'https://ballchasing.com/api/groups/?group={group_id}'  # API request URI
    headers = {'Authorization': token}
    try:
        request = requests.get(parent_group_uri, headers=headers)
        parent_page = request.json()

        try:
            parent_list = parent_page['list']
            parent_list_formatted = format_group(parent_list)
            return parent_list_formatted

        except KeyError:
            return format_group(parent_page)

    except json.decoder.JSONDecodeError:
        print("Error 500: ballchasing.com resources unavailable")
        return None


def get_replays_in_groups(group_id: str, token: str):
    """Do the get request to get replays in groups
    :param group_id: ballchasing.com group ID
    :param token: ballchasing.com API token
    :return: list of replays
    """
    parent_group_uri = f'https://ballchasing.com/api/replays/?group={group_id}'
    headers = {'Authorization': token}

    try:
        request = requests.get(parent_group_uri, headers=headers)
        replays = request.json()
        return replays

    except json.decoder.JSONDecodeError:
        print("Error 500: ballchasing.com resources unavailable")
        return None


def format_group(ballchasing_group_list: list):
    """Filter group to keep relevant details
    :param ballchasing_group_list: list of ballchasing.com groups
    :return: basic groups' infos.
    """
    return [{'id': group['id'],
             'name': group['name'],
             'link': group['link'].replace('api/groups', 'group'),
             'direct_replays': group['direct_replays'],
             'indirect_replays': group['indirect_replays']} for group in ballchasing_group_list]


def exploring_group(group_id: str, token: str):  # Works on RLStats group with Main Events only.
    """Browse into a ballchasing.com group (the highest one) with extra step for specific children groups / replays
    :param group_id: ballchasing.com group ID
    :param token: ballchasing.com API token
    :return replays: replays' ID in the parent group with region, split, event, phase, stage, round and match attached
    """
    replays = []
    rlcs_group = get_groups(group_id, token)

    for split in rlcs_group:  # Iteration on Splits.
        split_groups = get_groups(split['id'], token)
        split_hidden_replays = split['indirect_replays']

        try:  # Covering Worlds Finals group.
            split_name = split['name'].split(' - ')[1]

        except IndexError:
            split_name = 'Worlds'

        print(split_name)

        if split_name == 'Fall':  # Process for Fall Split
            replays += fall_routine(hidden_replays=split_hidden_replays,
                                    groups=split_groups,
                                    token=token)

        elif split_name == 'Winter':  # Process for Winter Split
            replays += winter_routine(hidden_replays=split_hidden_replays,
                                      groups=split_groups,
                                      token=token)

    return replays


def fall_routine(hidden_replays: int, groups: list, token: str):
    """Do the Fall Split events iteration process
    :param hidden_replays: split hidden replays
    :param groups: split subgroups
    :param token: ballchasing.com API token
    :return replays: replays' ID in the parent group with region, split, event, phase, stage, round and match attached
    """

    def fix_items(r_list: list):
        """Fix 'round' field in replays list ('Finals' and 'Semifinals' in place of 'top_4', etc.)
        :param r_list: original list of replays (list of dictionaries)
        :return: fixed list
        """
        with open('../../data/public/pre_patch.json', 'r', encoding='utf8') as pre_patch_file:
            pre_patch = json.load(pre_patch_file)

        for replay in r_list:
            if replay['round'] == 'top_4':
                if 'Series 1' in replay['match'] or 'Series 2' in replay['match']:
                    replay['round'] = 'Semifinal'

                else:
                    replay['round'] = 'Final'

            if replay['ballchasing_id'] in pre_patch.keys():  # Replacing wrong replays by good ones.
                replay['ballchasing_id'] = pre_patch[replay['ballchasing_id']]

        return r_list

    replays_list = []

    if hidden_replays > 0:  # Skip empty group.

        for event_type in groups:  # Iteration on event type (LAN / Online regionals, tiebreaker, etc.).
            event_type_groups = get_groups(event_type['id'], token)
            event_type_name = event_type['name']
            print(f'└── {event_type_name}')

            if event_type_name == 'International Major':  # Covering Major.
                region_name = 'World'
                event_name = 'Major'

                for stage in event_type_groups:  # Iteration on stage (Swiss stage then Playoffs).
                    stage_groups = get_groups(stage['id'], token)
                    stage_name = stage['name'].split(' - ')[1]
                    print(f'    └── {stage_name}')

                    for _round in stage_groups:  # Iteration on rounds (Round 1, 2, 3, ... ,Finals)
                        round_groups = get_groups(_round['id'], token)
                        round_name = _round['name']
                        print(f'        └── {round_name}')

                        for series in round_groups:  # Iteration on series.
                            series_name = series['name']

                            if round_name == 'Day 1':
                                round_name = 'Quarterfinal'

                            elif round_name == 'Day 2':
                                round_name = 'top_4'

                            print(f'            └── {series_name}')

                            series_group = get_groups(series['id'], token)  # Possible stats correction

                            if series_group:
                                for cor_group in series_group:
                                    sub_group_name = cor_group['name']

                                    print(f'                        └── * {sub_group_name}')

                                    # Addition to returned list.

                                    replays_list += [{"region": region_name,
                                                      "split": 'Fall',
                                                      "event": event_name,
                                                      "phase": 'Main Event',
                                                      "stage": stage_name,
                                                      "round": round_name,
                                                      "match": series_name,
                                                      "stats_correction": True,
                                                      "ballchasing_id": replay['id']} for replay in
                                                     get_replays_in_groups(cor_group["id"], token)["list"]]

                            # Addition to returned list.

                            replays_list += [{'region': region_name,
                                              'split': 'Fall',
                                              'event': event_name,
                                              'phase': 'Main Event',
                                              'stage': stage_name,
                                              'round': round_name,
                                              'match': series_name,
                                              "stats_correction": False,
                                              'ballchasing_id': replay['id']} for replay in
                                             get_replays_in_groups(series["id"], token)["list"]]

            else:
                for region in event_type_groups:  # Iteration on regions
                    region_groups = get_groups(region['id'], token)
                    region_name = region['name'].split(' - ')[1]

                    print(f'    └── {region_name}')

                    for event in region_groups:  # Iteration on events
                        event_groups = get_groups(event['id'], token)
                        event_name = event['name']

                        print(f'        └── {event_name}')

                        for stage in event_groups:  # Iteration on stage (Swiss stage, Playoffs).
                            stage_groups = get_groups(stage['id'], token)
                            stage_name = stage['name'].split(' - ')[1]

                            print(f'            └── {stage_name}')

                            for _round in stage_groups:  # Iteration on rounds (Round 1, 2, 3, ... ,Finals)
                                round_groups = get_groups(_round['id'], token)
                                round_name = _round['name']

                                print(f'                └── {round_name}')

                                for series in round_groups:  # Iteration on series.
                                    series_name = series['name']

                                    if round_name == 'Day 1':
                                        round_name = 'Quarterfinal'

                                    elif round_name == 'Day 2':
                                        round_name = 'top_4'

                                    print(f'                    └── {series_name}')

                                    series_group = get_groups(series['id'], token)  # Possible stats correction

                                    if series_group:
                                        for cor_group in series_group:
                                            sub_group_name = cor_group['name']

                                            print(f'                        └── * {sub_group_name}')

                                            # Addition to returned list.

                                            replays_list += [{"region": region_name,
                                                              "split": 'Fall',
                                                              "event": event_name,
                                                              "phase": 'Main Event',
                                                              "stage": stage_name,
                                                              "round": round_name,
                                                              "match": series_name,
                                                              "stats_correction": True,
                                                              "ballchasing_id": replay['id']} for replay in
                                                             get_replays_in_groups(cor_group["id"], token)["list"]]

                                    # Addition to returned list.

                                    replays_list += [{"region": region_name,
                                                      "split": 'Fall',
                                                      "event": event_name,
                                                      "phase": 'Main Event',
                                                      "stage": stage_name,
                                                      "round": round_name,
                                                      "match": series_name,
                                                      "stats_correction": False,
                                                      "ballchasing_id": replay['id']}
                                                     for replay in get_replays_in_groups(series["id"], token)["list"]]

    replays_list = fix_items(replays_list)

    return replays_list


def winter_routine(hidden_replays: int, groups: list, token: str):
    """Do the Fall Split events iteration process
    :param hidden_replays: split hidden replays
    :param groups: split subgroups
    :param token: ballchasing.com API token
    :return replays: replays' ID in the parent group with region, split, event, phase, stage, round and match attached
    """

    def fix_items(r_list: list):
        """Fix 'round' field in replays list ('Finals' and 'Semifinals' in place of 'Top4')
        :param r_list: original list of replays (list of dictionaries)
        :return: fixed list
        """
        with open('../../data/public/pre_patch.json', 'r', encoding='utf8') as pre_patch_file:
            pre_patch = json.load(pre_patch_file)

        for replay in r_list:
            if replay['round'] == 'top_4':
                if 'Series 1' in replay['match']:
                    replay['round'] = 'Upper Final'

                elif 'Series 2' in replay['match']:
                    replay['round'] = 'Lower Semifinal'

                elif 'Series 3' in replay['match']:
                    replay['round'] = 'Lower Final'

                else:
                    replay['round'] = 'Grand Final'

            elif replay['round'] == 'group_match':
                if 'Series 1' in replay['match'] or 'Series 2' in replay['match']:
                    replay['round'] = 'Round 1'

                elif 'Series 3' in replay['match'] or 'Series 4' in replay['match']:
                    replay['round'] = 'Round 2'

                elif 'Series 5' in replay['match'] or 'Series 6' in replay['match']:
                    replay['round'] = 'Round 3'

                else:
                    replay['round'] = 'Tiebreaker Round'

            else:
                if 'Series 1' in replay['match'] or 'Series 2' in replay['match']:
                    replay['round'] = 'Lower Round 2'

                elif 'Series 3' in replay['match'] or 'Series 4' in replay['match']:
                    replay['round'] = 'Upper Semifinal'

                elif 'Series 5' in replay['match'] or 'Series 6' in replay['match']:
                    replay['round'] = 'Lower Quarterfinal'

                else:
                    replay['round'] = 'Tiebreaker Round'

            if replay['ballchasing_id'] in pre_patch.keys():
                replay['ballchasing_id'] = pre_patch[replay['ballchasing_id']]

        return r_list

    replays_list = []
    if hidden_replays > 0:  # Skip empty group.

        for event_type in groups:  # Iteration on event type (LAN / Online regionals, tiebreaker, etc.).
            event_type_groups = get_groups(event_type['id'], token)
            event_type_name = event_type['name']

            print(f'└── {event_type_name}')

            if event_type_name == 'International Major':  # Covering Major.
                pass  # No event at the moment
                # region_name = 'World'
                # event_name = 'Major'

            else:
                for region in event_type_groups:  # Iteration on regions
                    region_groups = get_groups(region['id'], token)
                    region_name = region['name'].split(' - ')[1]

                    print(f'    └── {region_name}')

                    for event in region_groups:  # Iteration on events
                        event_groups = get_groups(event['id'], token)
                        event_name = event['name']

                        print(f'        └── {event_name}')

                        for stage in event_groups:  # Iteration on stage (Group stage then Playoffs).
                            stage_groups = get_groups(stage['id'], token)
                            stage_name = stage['name'].split(' - ')[1]

                            print(f'            └── {stage_name}')

                            for group_or_day in stage_groups:  # Iteration on rounds (Round 1, 2, 3, ... ,Finals)
                                group_or_day_groups = get_groups(group_or_day['id'], token)
                                group_or_day_name = group_or_day['name']

                                print(f'                └── {group_or_day_name}')

                                if group_or_day_groups:
                                    for series in group_or_day_groups:  # Iteration on series.
                                        series_name = series['name']

                                        print(f'                    └── {series_name}')

                                        round_name = None

                                        if "Group" in group_or_day_name:
                                            round_name = "group_match"

                                        elif "Day 1" in group_or_day_name:
                                            round_name = "Lower Round 1"

                                        elif "Day 2" in group_or_day_name:
                                            round_name = "top_8"

                                        elif "Day 3" in group_or_day_name:
                                            round_name = "top_4"

                                        series_group = get_groups(series['id'], token)  # Possible stats correction

                                        if series_group:
                                            for cor_group in series_group:
                                                sub_group_name = cor_group['name']

                                                print(f'                        └── * {sub_group_name}')

                                                # Addition to returned list.

                                                replays_list += [{"region": region_name,
                                                                  "split": 'Winter',
                                                                  "event": event_name,
                                                                  "phase": 'Main Event',
                                                                  "stage": stage_name,
                                                                  "round": round_name,
                                                                  "match": series_name,
                                                                  "stats_correction": True,
                                                                  "ballchasing_id": replay['id']} for replay in
                                                                 get_replays_in_groups(cor_group["id"], token)["list"]]

                                        # Addition to returned list.

                                        replays_list += [{"region": region_name,
                                                          "split": 'Winter',
                                                          "event": event_name,
                                                          "phase": 'Main Event',
                                                          "stage": stage_name,
                                                          "round": round_name,
                                                          "match": series_name,
                                                          "stats_correction": False,
                                                          "ballchasing_id": replay['id']}
                                                         for replay in
                                                         get_replays_in_groups(series["id"], token)["list"]]

    replays_list = fix_items(replays_list)

    return replays_list


def get_replay_stats(replay: dict, token: str):
    """Get detailed replay's stats
    :param replay: basic information from a replay uploaded on ballchasing.com (at least the ballchasing_id)
    :param token: ballchasing.com API token
    :return replay_stats: detailed replay stat as a dict
    """
    replay_id = replay['ballchasing_id']
    replay_uri = f'https://ballchasing.com/api/replays/{replay_id}'
    headers = {'Authorization': token}

    try:
        request = requests.get(replay_uri, headers=headers)
        replay_stats = request.json()

        replay_stats.pop('id')
        replay_stats['link'] = replay_stats['link'].replace('api/replays', 'replay')

        try:
            for group in replay_stats['groups']:
                group['link'] = group['link'].replace('api/groups', 'group')

        except KeyError:
            pass

        replay['details'] = replay_stats

        return replay

    except json.decoder.JSONDecodeError:
        print("Error 500: Resources unavailable")
        return None


def add_details(replays_unfiltered: list, raw_list: list, token: str, workers: int = 8):
    """Add detailed stats to replays
    :param replays_unfiltered: list of all replays without details
    :param raw_list: list of replay already retrieved
    :param token: ballchasing.com API token
    :param workers: CPU used for multiprocessing, must correspond to the number of possible requests per second (8
    with Patreon Tier 3 support)
    :return replay_list: list of replays with details added into 'details' field
    """
    try:
        raw_ids = {replay['ballchasing_id'] for replay in raw_list}
        replays_filtered = [replay for replay in replays_unfiltered if replay['ballchasing_id'] not in raw_ids]
        replay_list = p_map(get_replay_stats, replays_filtered, [token] * len(replays_filtered), num_cpus=workers)

    except TypeError:
        raw_list = []
        replay_list = p_map(get_replay_stats, replays_unfiltered, [token] * len(replays_unfiltered), num_cpus=workers)

    return replay_list + raw_list
