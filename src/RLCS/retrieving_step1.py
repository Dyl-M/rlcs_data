# -*- coding: utf-8 -*-

import json
import requests

from time import sleep

"""File Informations

@file_name: retrieving_step1.py
@author: Dylan "dyl-m" Monfret
"""

"PREPARATORY ELEMENTS"

with open('../../data/private/my_token.txt', 'r', encoding='utf8') as token_file:
    my_token = token_file.read()

RLCS_EU_NA = 'rlcs-21-22-t30czbd3he'
# RLCS_OCE = ''
# RLCS_SAM = ''
# RLCS_MENA = ''
# RLCS_APAC = ''
# RLCS_SSA = ''

"FUNCTIONS"


def get_groups(group_id, token):
    """Get children groups and direct replays summaries from a ballchasing.com parent groups.

    :param group_id: ballchasing.com group ID.
    :param token: ballchasing.com API token.
    :return: children groups and direct replays summaries.
    """
    parent_group_url = f'https://ballchasing.com/api/groups/?group={group_id}'
    headers = {'Authorization': token}
    try:
        request = requests.get(parent_group_url, headers=headers)
        parent_page = request.json()

        try:
            parent_list = parent_page['list']
            parent_list_formatted = format_group(parent_list)
            sleep(1)
            return parent_list_formatted

        except KeyError:
            sleep(1)
            return format_group(parent_page)

    except json.decoder.JSONDecodeError:
        print("Error 500: Resources unavailable")
        return None


def get_replays_in_groups(group_id, token):
    """Do the get request to get replays in groups.

    :param group_id: ballchasing.com group ID.
    :param token: ballchasing.com API token.
    :return: list of replays.
    """
    parent_group_url = f'https://ballchasing.com/api/replays/?group={group_id}'
    headers = {'Authorization': token}

    try:
        request = requests.get(parent_group_url, headers=headers)
        replays = request.json()
        sleep(1)
        return replays

    except json.decoder.JSONDecodeError:
        print("Error 500: Resources unavailable")
        return None


def format_group(ballchasing_group_list):
    """Filter group's informations.

    :param ballchasing_group_list:
    :return: basic groups' informations.
    """
    return [{'id': group['id'],
             'name': group['name'],
             'link': group['link'].replace('api/groups', 'group'),
             'direct_replays': group['direct_replays'],
             'indirect_replays': group['indirect_replays']} for group in ballchasing_group_list]


def exploring_group(group_id, token):
    """Browse into a ballchasing.com group (highest one) with extra step for specific children groups / replays.

    :param group_id: ballchasing.com group ID.
    :param token: ballchasing.com API token.
    :return: all replays ID in the parent group with region, split, event, phase, stage, round and match attached.
    """
    replays = []
    rlcs_group = get_groups(group_id, token)

    for split in rlcs_group:
        split_groups = get_groups(split['id'], token)

        for event in split_groups:
            event_groups = get_groups(event['id'], token)

            for region in event_groups:
                region_groups = get_groups(region['id'], token)

                for phase in region_groups:
                    if phase['direct_replays'] == phase['indirect_replays']:
                        replays += [{"region": None,
                                     "split": split['name'],
                                     "event": event['name'],
                                     "phase": region['name'],
                                     "stage": None,
                                     "round": None,
                                     "match": phase['name'],
                                     "ballchasing_id": replay['id']}
                                    for replay in get_replays_in_groups(phase["id"], token)["list"]]

                    phase_groups = get_groups(phase['id'], token)

                    for stage in phase_groups:
                        if stage['direct_replays'] == stage['indirect_replays']:
                            replays += [{"region": region['name'],
                                         "split": split['name'],
                                         "event": event['name'],
                                         "phase": phase['name'],
                                         "stage": None,
                                         "round": None,
                                         "match": stage['name'],
                                         "ballchasing_id": replay['id']}
                                        for replay in get_replays_in_groups(stage["id"], token)["list"]]

                        stage_groups = get_groups(stage['id'], token)

                        for _round in stage_groups:
                            if _round['direct_replays'] == _round['indirect_replays']:
                                replays += [{"region": region['name'],
                                             "split": split['name'],
                                             "event": event['name'],
                                             "phase": "Main Event",
                                             "stage": phase['name'],
                                             "round": stage['name'],
                                             "match": _round['name'],
                                             "ballchasing_id": replay['id']}
                                            for replay in get_replays_in_groups(_round["id"], token)["list"]]

                            round_groups = get_groups(_round['id'], token)

                            for match in round_groups:
                                replays += [{"region": region['name'],
                                             "split": split['name'],
                                             "event": event['name'],
                                             "phase": phase['name'],
                                             "stage": stage['name'],
                                             "round": _round['name'],
                                             "match": match['name'],
                                             "ballchasing_id": replay['id']}
                                            for replay in get_replays_in_groups(match["id"], token)["list"]]
    return replays


def fill_none(pre_made_replays_list):
    """Fill none and reformat some fields in a pre-made replays list.

    :param pre_made_replays_list: pre-made replays list from ballchasing.com.
    :return: formatted list.
    """
    for replay in pre_made_replays_list:
        if replay["region"] is None:
            replay["region"] = "North America"
            replay["phase"] = "Qualifier"
            replay["stage"] = "Tiebreaker"
            replay["round"] = "Finals"

        elif replay['region'] == 'Europe' and replay['phase'] == 'Tiebreaker':
            replay["phase"] = "Qualifier"
            replay["stage"] = "Tiebreaker"

            if replay["match"] == "EG vs 00":
                replay["round"] = "Lower Finals"

            else:
                replay["round"] = "Upper Finals"

    return pre_made_replays_list


"MAIN"

if __name__ == '__main__':
    rlcs_replays = exploring_group(RLCS_EU_NA, my_token)
    the_none = fill_none(rlcs_replays)
    with open('../../data/retrieved/pre_dataset.json', 'w', encoding='utf-8') as pre_dataset_file:
        json.dump(rlcs_replays, pre_dataset_file)
