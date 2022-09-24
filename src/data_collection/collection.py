# -*- coding: utf-8 -*-

import datetime
import itertools
import json
import numpy as np
import p_tqdm
import pandas as pd
import re
import requests
import sys
import time
import tqdm
import tzlocal

"""File Information
@file_name: collection.py
@author: Dylan "dyl-m" Monfret
"""

"OPTIONS"

pd.set_option('display.max_columns', None)  # pd.set_option('display.max_rows', None)
pd.set_option('display.width', 225)

"GLOBAL"

rlcs2122 = 'rlcs2122'
TODAY = datetime.datetime.now(tz=tzlocal.get_localzone())

with open('../../data/private/my_token.txt', 'r', encoding='utf8') as token_file:
    my_token = token_file.read()

with open('../../data/public/true_cols.json', 'r', encoding='utf8') as true_cols_file:
    true_cols = json.load(true_cols_file)

with open('../../data/public/missing_value.json', 'r', encoding='utf8') as missing_value_file:
    missing_value = json.load(missing_value_file)

with open('../../data/public/winter_major_patch.json', 'r', encoding='utf8') as winter_major_patch_file:
    winter_major_patch = json.load(winter_major_patch_file)

try:
    with open('../../data/retrieved/replays_tmp.json', 'r', encoding='utf8') as replays_tmp_file:
        replays_tmp = json.load(replays_tmp_file)

except FileNotFoundError:
    replays_tmp = []

"FUNCTIONS"


def format_string_underscore(col_string):
    """Format a character string in lower case with underscore as separator
    :param col_string: column name to parse
    :return: lower case string with underscore as separation.
    """
    return "_".join(re.split('(?=[A-Z])', col_string)).lower()


def find_missing_players_country(players_df: pd.DataFrame):
    """Find and save players without assigned country
    :param players_df: a dataframe with player identifier.
    """
    country_miss = players_df.loc[players_df.player_country.isna(),
                                  ['player_id', 'player_tag', 'player_country']].drop_duplicates()

    country_miss.player_tag = country_miss.player_tag.str.lower()
    country_miss.sort_values(['player_tag'], inplace=True)
    country_miss.to_json('../../data/public/miss_country_tmp.json', orient="records", indent=1)


def missing_map_name_id(dataframe: pd.DataFrame):
    """Add missing map names to dataframe
    :param dataframe: main / original pandas dataframe
    :return: fixed pandas dataframe.
    """
    for rl_map_name in missing_value['map_names']:
        map_name = rl_map_name['map_name']
        map_id = rl_map_name['map_id']
        dataframe.loc[dataframe['map_id'] == map_id, 'map_name'] = map_name

    for rl_map_id in missing_value['map_ids']:
        map_name = rl_map_id['map_name']
        map_id = rl_map_id['map_id']
        dataframe.loc[dataframe['map_name'] == map_name, 'map_id'] = map_id

    return dataframe


def missing_player_country(dataframe: pd.DataFrame):
    """Add missing player countries to dataframe
    :param dataframe: main / original pandas dataframe
    :return: fixed pandas dataframe.
    """
    for player in missing_value['players_country']:
        player_id = player['player_id']
        player_country = player['player_country']
        dataframe.loc[dataframe['player_id'] == player_id, 'player_country'] = player_country

    return dataframe


def missing_car_name(dataframe: pd.DataFrame):
    """Add missing car names to dataframe
    :param dataframe: main / original pandas dataframe
    :return: fixed pandas dataframe.
    """
    for car in missing_value['car_names']:
        car_name = car['car_name']
        car_id = car['car_id']
        dataframe.loc[dataframe['car_id'] == car_id, 'car_name'] = car_name

    return dataframe


def format_teams_df(ref_df: pd.DataFrame, neut_cols: list, side_cols: list, color: str):
    """Format a team dataframe (avoid code duplication)
    :param ref_df: a reference dataframe (often a dataframe per match or game)
    :param neut_cols: neutral columns as a list
    :param side_cols: side columns (orange / blue team) as a list
    :param color: team color (blue or orange)
    :return teams_df: a formatted dataframe with information by team.
    """
    teams_df = pd.concat([ref_df.loc[:, neut_cols],
                          pd.DataFrame({'color': []}),
                          ref_df.loc[:, side_cols]], axis=1)

    teams_df['color'] = color  # Set color

    # Remove columns prefix
    teams_df.columns = teams_df.columns.str.replace(f'{color}_team_', '')
    teams_df.columns = teams_df.columns.str.replace(f'{color}_', '')

    return teams_df


def format_players_df(ref_df: pd.DataFrame, neut_cols: list, side_cols: list, color: str):
    """Format a team dataframe (avoid code duplication)
    :param ref_df: a reference dataframe (often a dataframe per match or game)
    :param neut_cols: neutral columns as a list
    :param side_cols: side columns (orange / blue team) as a list
    :param color: team color (blue or orange)
    :return players_df: a formatted dataframe with information by player.
    """
    players_df = pd.json_normalize(pd.concat([ref_df.loc[:, neut_cols + side_cols],
                                              pd.DataFrame({'color': []}),
                                              ref_df.loc[:, [f'{color}_players', f'{color}_winner']]], axis=1)
                                   .explode(f'{color}_players')
                                   .to_dict('records'), sep='_').dropna(axis=1, how='all')

    players_df['color'] = color  # Set color

    # Remove columns prefix
    players_df.columns = players_df.columns.str.replace(f'{color}_team_', '')
    players_df.columns = players_df.columns.str.replace(f'{color}_players_', '')
    players_df.columns = players_df.columns.str.replace(f'{color}_', '')

    return players_df


def get_an_event(events_id: str):
    """Get a single event with octane.gg event slug
    :param events_id: event group string
    :return: events ID list.
    """
    uri = f'https://zsr.octane.gg/events/{events_id}'

    try:
        events_list = requests.get(uri).json()
        return events_list

    except json.decoder.JSONDecodeError:
        print('json.decoder.JSONDecodeError')
        sys.exit()


def get_events(events_group: str):
    """Get event list with octane.gg group identifier
    :param events_group: event group string
    :return: events ID list.
    """
    uri = f'https://zsr.octane.gg/events?group={events_group}'

    try:
        events_list = requests.get(uri).json()['events']
        return events_list

    except json.decoder.JSONDecodeError:
        print('json.decoder.JSONDecodeError')
        sys.exit()


def get_event_matches(event_id: str):
    """Get games list with octane.gg event identifier
    :param event_id: octane.gg event ID
    :return: event games list.
    """
    uri = f'https://zsr.octane.gg/matches?event={event_id}'

    try:
        matches_list = [match for match in requests.get(uri).json()['matches'] if 'tbd' not in match['slug']]
        return matches_list

    except json.decoder.JSONDecodeError:
        print('json.decoder.JSONDecodeError')
        sys.exit()


def get_game_details(game_id: str):
    """Get games list with octane.gg event identifier
    :param game_id: octane.gg game ID
    :return: games details.
    """
    uri = f'https://zsr.octane.gg/games/{game_id}'

    try:
        matches_list = requests.get(uri).json()
        return matches_list

    except json.decoder.JSONDecodeError:
        print('json.decoder.JSONDecodeError')
        sys.exit()


def get_player(player_id: str):
    """Get player detailed information with octane.gg player identifier
    :param player_id: octane.gg player ID
    :return: games details.
    """
    uri = f'https://zsr.octane.gg/players/{player_id}'

    try:
        player_info = requests.get(uri).json()
        return player_info

    except json.decoder.JSONDecodeError:
        print('json.decoder.JSONDecodeError')
        sys.exit()


def get_ballchasing(ballchasing_id: str, token: str = my_token):
    """Get a ballchasing replay details
    :param ballchasing_id: ballchasing.com replay ID
    :param token: ballchasing.com API token
    :return replay: returns the desired items associated with the replay.
    """
    replay_uri = f'https://ballchasing.com/api/replays/{ballchasing_id}'
    headers = {'Authorization': token}
    ballchasing_request = requests.get(replay_uri, headers=headers)
    status_code = ballchasing_request.status_code

    def perform_request(request):
        """Extract required information from a ballchasing.com API call
        :param request: request results (response is valid / 200)
        :return replay: dictionary with required information (settings, car, platform information).
        """
        response = request.json()
        replay = {'id': response['id'], 'players': []}

        try:
            for color in ['blue', 'orange']:
                for a_player in response[color]['players']:
                    if all(key in a_player.keys() for key in ['car_name', 'car_id']):
                        replay['players'].append({'accounts': a_player['id'], 'camera': a_player['camera'],
                                                  'steering_sensitivity': a_player['steering_sensitivity'],
                                                  'car_id': str(a_player['car_id']), 'car_name': a_player['car_name']})
                    elif 'car_id' in a_player.keys():
                        replay['players'].append({'accounts': a_player['id'], 'camera': a_player['camera'],
                                                  'steering_sensitivity': a_player['steering_sensitivity'],
                                                  'car_id': str(a_player['car_id']), 'car_name': None})
                    else:
                        replay['players'].append({'accounts': a_player['id'], 'camera': a_player['camera'],
                                                  'steering_sensitivity': a_player['steering_sensitivity'],
                                                  'car_id': None, 'car_name': None})

            return replay

        except json.decoder.JSONDecodeError:
            print('json.decoder.JSONDecodeError')
            sys.exit()

    if status_code == 200:
        return perform_request(request=ballchasing_request)

    if status_code == 404:
        return {'id': ballchasing_id}

    if status_code == 429:
        time.sleep(1)
        return {'error': ballchasing_id}

    print(f'UNKNOWN ERROR: {status_code} / {ballchasing_id}')
    sys.exit()


def add_rounds(matches_df: pd.DataFrame):
    """Add round information to a matches dataframe
    :param matches_df: matches pandas dataframe
    :return matches_df: enhanced dataframe.
    """
    rounds_df = matches_df.loc[:, ['event_id', 'event_split', 'event', 'event_phase', 'stage', 'match_number',
                                   'match_id']].drop_duplicates()

    cond_swiss_round_1 = (rounds_df.stage == 'Swiss Stage') & (rounds_df.match_number.between(1, 8))
    cond_swiss_round_2 = (rounds_df.stage == 'Swiss Stage') & (rounds_df.match_number.between(9, 16))
    cond_swiss_round_3 = (rounds_df.stage == 'Swiss Stage') & (rounds_df.match_number.between(17, 24))
    cond_swiss_round_4 = (rounds_df.stage == 'Swiss Stage') & (rounds_df.match_number.between(25, 30))
    cond_swiss_round_5 = (rounds_df.stage == 'Swiss Stage') & (rounds_df.match_number > 30)

    cond_winter_group_round_1 = (rounds_df.event_split == 'Winter') & (rounds_df.stage == 'Group Stage') & \
                                (rounds_df.match_number.isin([1, 2, 7, 8, 13, 14, 19, 20]))

    cond_winter_group_round_2 = (rounds_df.event_split == 'Winter') & (rounds_df.stage == 'Group Stage') & \
                                (rounds_df.match_number.isin([3, 4, 9, 10, 15, 16, 21, 22]))

    cond_winter_group_round_3 = (rounds_df.event_split == 'Winter') & (rounds_df.stage == 'Group Stage') & \
                                (rounds_df.match_number.isin([5, 6, 11, 12, 17, 18, 23, 24]))

    cond_winter_group_tb = (rounds_df.event_split == 'Winter') & (rounds_df.stage == 'Group Stage') & \
                           (rounds_df.match_number > 24)

    cond_fall_playoff_qf = (rounds_df.event_split == 'Fall') & (rounds_df.stage == 'Playoffs') & \
                           (rounds_df.match_number.between(1, 4))

    cond_fall_playoff_sf = (rounds_df.event_split == 'Fall') & (rounds_df.stage == 'Playoffs') & \
                           (rounds_df.match_number.between(5, 6))

    cond_fall_playoff_fn = (rounds_df.event_split == 'Fall') & (rounds_df.stage == 'Playoffs') & \
                           (rounds_df.match_number > 6)

    cond_winter_playoff_lr1 = (rounds_df.event_split == 'Winter') & (rounds_df.stage == 'Playoffs') & \
                              (rounds_df.match_number.between(4, 7))

    cond_winter_playoff_lr2 = (rounds_df.event_split == 'Winter') & (rounds_df.stage == 'Playoffs') & \
                              (rounds_df.match_number.between(8, 9))

    cond_winter_playoff_usf = (rounds_df.event_split == 'Winter') & (rounds_df.stage == 'Playoffs') & \
                              (rounds_df.match_number.between(1, 2))

    cond_winter_playoff_lqf = (rounds_df.event_split == 'Winter') & (rounds_df.stage == 'Playoffs') & \
                              (rounds_df.match_number.between(10, 11))

    cond_winter_playoff_ufn = (rounds_df.event_split == 'Winter') & (rounds_df.stage == 'Playoffs') & \
                              (rounds_df.match_number == 3)

    cond_winter_playoff_lsf = (rounds_df.event_split == 'Winter') & (rounds_df.stage == 'Playoffs') & \
                              (rounds_df.match_number == 12)

    cond_winter_playoff_lfn = (rounds_df.event_split == 'Winter') & (rounds_df.stage == 'Playoffs') & \
                              (rounds_df.match_number == 13)

    cond_winter_playoff_gfn = (rounds_df.event_split == 'Winter') & (rounds_df.stage == 'Playoffs') & \
                              (rounds_df.match_number > 13)

    cond_spring_playoff_ur1 = (rounds_df.event_split == 'Spring') & (rounds_df.stage == 'Playoffs') & \
                              (rounds_df.match_number.between(1, 8))

    cond_spring_playoff_uqf = (rounds_df.event_split == 'Spring') & (rounds_df.stage == 'Playoffs') & \
                              (rounds_df.match_number.between(9, 12))

    cond_spring_playoff_usf = (rounds_df.event_split == 'Spring') & (rounds_df.stage == 'Playoffs') & \
                              (rounds_df.match_number.between(13, 14))

    cond_spring_playoff_ufn = (rounds_df.event_split == 'Spring') & (rounds_df.stage == 'Playoffs') & \
                              (rounds_df.match_number == 15)

    cond_spring_playoff_lr1 = (rounds_df.event_split == 'Spring') & (rounds_df.stage == 'Playoffs') & \
                              (rounds_df.match_number.between(16, 19))

    cond_spring_playoff_lr2 = (rounds_df.event_split == 'Spring') & (rounds_df.stage == 'Playoffs') & \
                              (rounds_df.match_number.between(20, 23))

    cond_spring_playoff_lr3 = (rounds_df.event_split == 'Spring') & (rounds_df.stage == 'Playoffs') & \
                              (rounds_df.match_number.between(24, 25))

    cond_spring_playoff_lqf = (rounds_df.event_split == 'Spring') & (rounds_df.stage == 'Playoffs') & \
                              (rounds_df.match_number.between(26, 27))

    cond_spring_playoff_lsf = (rounds_df.event_split == 'Spring') & (rounds_df.stage == 'Playoffs') & \
                              (rounds_df.match_number == 28)

    cond_spring_playoff_lfn = (rounds_df.event_split == 'Spring') & (rounds_df.stage == 'Playoffs') & \
                              (rounds_df.match_number == 29)

    cond_spring_playoff_gfn = (rounds_df.event_split == 'Spring') & (rounds_df.stage == 'Playoffs') & \
                              (rounds_df.match_number > 29)

    cond_worlds_wildcard_r1 = (rounds_df.stage == 'Wildcard') & (rounds_df.match_number.between(1, 8))
    cond_worlds_wildcard_r2 = (rounds_df.stage == 'Wildcard') & (rounds_df.match_number.between(9, 16))
    cond_worlds_wildcard_r3 = (rounds_df.stage == 'Wildcard') & (rounds_df.match_number.between(17, 24))
    cond_worlds_wildcard_r4 = (rounds_df.stage == 'Wildcard') & (rounds_df.match_number.between(25, 30))
    cond_worlds_wildcard_r5 = (rounds_df.stage == 'Wildcard') & (rounds_df.match_number > 30)

    cond_worlds_group_ur1 = (rounds_df.event_split == 'Summer') & (rounds_df.stage == 'Group Stage') & \
                            (rounds_df.match_number.isin([1, 2, 3, 4, 11, 12, 13, 14]))

    cond_worlds_group_ur2 = (rounds_df.event_split == 'Summer') & (rounds_df.stage == 'Group Stage') & \
                            (rounds_df.match_number.isin([5, 6, 15, 16]))

    cond_worlds_group_lr1 = (rounds_df.event_split == 'Summer') & (rounds_df.stage == 'Group Stage') & \
                            (rounds_df.match_number.isin([7, 8, 17, 18]))

    cond_worlds_group_lr2 = (rounds_df.event_split == 'Summer') & (rounds_df.stage == 'Group Stage') & \
                            (rounds_df.match_number.isin([9, 10, 19, 20]))

    cond_worlds_playoff_qf = (rounds_df.event_split == 'Summer') & (rounds_df.stage == 'Playoffs') & \
                             (rounds_df.match_number.between(1, 4))

    cond_worlds_playoff_sf = (rounds_df.event_split == 'Summer') & (rounds_df.stage == 'Playoffs') & \
                             (rounds_df.match_number.between(5, 6))

    cond_worlds_playoff_fn = (rounds_df.event_split == 'Summer') & (rounds_df.stage == 'Playoffs') & \
                             (rounds_df.match_number > 6)

    cond_apac_usf = (rounds_df.event_split != 'Spring') & (rounds_df.event == 'APAC Qualifier') & \
                    (rounds_df.match_number.between(1, 2))

    cond_apac_lsf = (rounds_df.event_split != 'Spring') & (rounds_df.event == 'APAC Qualifier') & \
                    (rounds_df.match_number == 3)

    cond_apac_ufn = (rounds_df.event_split != 'Spring') & (rounds_df.event == 'APAC Qualifier') & \
                    (rounds_df.match_number == 4)

    cond_apac_lfn = (rounds_df.event_split != 'Spring') & (rounds_df.event == 'APAC Qualifier') & \
                    (rounds_df.match_number == 5)

    cond_apac_gfn = (rounds_df.event_split != 'Spring') & (rounds_df.event == 'APAC Qualifier') & \
                    (rounds_df.match_number > 5)

    cond_apac_sp_ufn = (rounds_df.event_split == 'Spring') & (rounds_df.event == 'APAC Qualifier') & \
                       (rounds_df.match_number == 1)

    cond_apac_sp_lsf = (rounds_df.event_split == 'Spring') & (rounds_df.event == 'APAC Qualifier') & \
                       (rounds_df.match_number == 2)

    cond_apac_sp_lfn = (rounds_df.event_split == 'Spring') & (rounds_df.event == 'APAC Qualifier') & \
                       (rounds_df.match_number == 3)

    cond_apac_sp_gf = (rounds_df.event_split == 'Spring') & (rounds_df.event == 'APAC Qualifier') & \
                      (rounds_df.match_number > 3)

    cond_tb = (rounds_df.event == 'Major Tiebreaker') | (rounds_df.event_phase == 'Qualifier Tiebreaker') | \
              (rounds_df.event_phase == 'Main Event Tiebreaker')

    rounds_df.loc[cond_swiss_round_1 | cond_winter_group_round_1 | cond_worlds_wildcard_r1, 'match_round'] = 'Round 1'
    rounds_df.loc[cond_swiss_round_2 | cond_winter_group_round_2 | cond_worlds_wildcard_r2, 'match_round'] = 'Round 2'
    rounds_df.loc[cond_swiss_round_3 | cond_winter_group_round_3 | cond_worlds_wildcard_r3, 'match_round'] = 'Round 3'
    rounds_df.loc[cond_swiss_round_4 | cond_worlds_wildcard_r4, 'match_round'] = 'Round 4'
    rounds_df.loc[cond_swiss_round_5 | cond_worlds_wildcard_r5, 'match_round'] = 'Round 5'
    rounds_df.loc[cond_winter_group_tb, 'match_round'] = 'Tiebreaker Round'
    rounds_df.loc[cond_fall_playoff_qf | cond_worlds_playoff_qf, 'match_round'] = 'Quarterfinal'
    rounds_df.loc[cond_fall_playoff_sf | cond_worlds_playoff_sf, 'match_round'] = 'Semifinal'

    rounds_df.loc[cond_winter_playoff_lr1 | cond_spring_playoff_lr1 | cond_worlds_group_lr1,
                  'match_round'] = 'Lower Round 1'

    rounds_df.loc[cond_winter_playoff_lr2 | cond_spring_playoff_lr2 | cond_worlds_group_lr2,
                  'match_round'] = 'Lower Round 2'

    rounds_df.loc[cond_spring_playoff_lr3, 'match_round'] = 'Lower Round 3'
    rounds_df.loc[cond_winter_playoff_lqf | cond_spring_playoff_lqf, 'match_round'] = 'Lower Quarterfinal'

    rounds_df.loc[cond_winter_playoff_lsf | cond_spring_playoff_lsf | cond_apac_lsf | cond_apac_sp_lsf,
                  'match_round'] = 'Lower Semifinal'

    rounds_df.loc[cond_winter_playoff_lfn | cond_spring_playoff_lfn | cond_apac_lfn | cond_apac_sp_lfn,
                  'match_round'] = 'Lower Final'

    rounds_df.loc[cond_spring_playoff_ur1 | cond_worlds_group_ur1, 'match_round'] = 'Upper Round 1'
    rounds_df.loc[cond_worlds_group_ur2, 'match_round'] = 'Upper Round 2'
    rounds_df.loc[cond_spring_playoff_uqf, 'match_round'] = 'Upper Quarterfinal'

    rounds_df.loc[cond_winter_playoff_usf | cond_spring_playoff_usf | cond_apac_usf | cond_apac_sp_ufn,
                  'match_round'] = 'Upper Semifinal'

    rounds_df.loc[cond_winter_playoff_ufn | cond_spring_playoff_ufn | cond_apac_ufn, 'match_round'] = 'Upper Final'

    rounds_df.loc[cond_winter_playoff_gfn | cond_spring_playoff_gfn | cond_apac_gfn | cond_apac_sp_gf,
                  'match_round'] = 'Grand Final'

    rounds_df.loc[cond_fall_playoff_fn | cond_tb | cond_worlds_playoff_fn, 'match_round'] = 'Final'

    matches_df = rounds_df.merge(matches_df)

    for a_match_round, match_id_list in winter_major_patch.items():  # Apply Winter Major patch, following SRG withdraw
        for a_match_id in match_id_list:
            matches_df.loc[matches_df.match_id == a_match_id, 'match_round'] = a_match_round

    return matches_df


def parse_events(events_group: str, event_id_list: list = None, csv_export: bool = False):
    """Parse event and "stages"
    :param events_group: event group string
    :param event_id_list: list of event identifier (char. string)
    :param csv_export: to export the final dataframe as .csv file or not
    :return: event dataframe.
    """
    if event_id_list is None:  # If one or several events are not in events_group yet
        events_list = get_events(events_group=events_group)

    else:
        events_list = get_events(events_group=events_group) + [get_an_event(event_id) for event_id in event_id_list]

    # First parsing and drop irrelevant columns
    events_df = pd.json_normalize(events_list, sep='_').drop(['image', 'groups', 'mode', 'prize_currency'], axis=1)

    # Second parsing: stages and drop irrelevant columns
    stages_df = pd.json_normalize(events_df.loc[:, ['_id', 'stages']].explode('stages').to_dict('records'), sep='_')
    stages_df.drop(['stages_region', 'stages_prize_currency'], axis=1, inplace=True)

    # Drop 'stages' columns parsed before and redundant 'prize_amount'
    events_df.drop(['stages', 'prize_amount'], axis=1, inplace=True)
    events_df = events_df.merge(stages_df).rename(columns=true_cols['renaming']['events'])  # Merge both dataframe
    events_df.prize_money.fillna(0, inplace=True)  # Fill NaN prize_money with 0
    events_df.stage_is_lan.fillna(False, inplace=True)  # Fill NaN stage_is_lan with False

    # Fill stage_is_qualifier with False and fix Major Tiebreaker and APAC Qualifiers values
    events_df.loc[events_df.event_tier == 'Qualifier', 'stage_is_qualifier'] = True
    events_df.stage_is_qualifier.fillna(False, inplace=True)

    # Create simplified 'event' column from event_name and replace values for Majors
    events_df['event'] = events_df.event_name.apply(lambda x: " ".join(x.split(" ")[-2:]))
    events_df.event.replace({'Fall Major': 'Major', 'Winter Major': 'Major', 'Spring Major': 'Major'}, inplace=True)

    # Create 'split' column from event_name and replace values for Worlds
    events_df['event_split'] = events_df.event_name.apply(lambda x: "".join(x.split(" ")[2]))
    events_df.event_split.replace('World', 'Summer', inplace=True)

    # Fix event region and stages
    events_df.event_region = events_df.event_name.apply(lambda x: " ".join(x.split(" ")[3:-2]))

    l_event_closed_tb = ['614b7804f8090ec74528643b',
                         '615354d5f8090ec745286981',
                         '614b7f42f8090ec745286447']  # Event with tiebreaker match for a place in Closed Qualifier
    l_event_main_tb = ['614b7975f8090ec74528643d']  # Event with a tiebreaker for a place in Main Event

    # Conditions for event Tiebreakers
    cond_closed_tb = (events_df.stage == 'Tiebreaker') & (events_df.event_id.isin(l_event_closed_tb))
    cond_main_tb = (events_df.stage == 'Tiebreaker') & (events_df.event_id.isin(l_event_main_tb))

    events_df.loc[events_df.event_region == '', 'event_region'] = 'World'
    events_df.loc[events_df.event_region == 'Major', ['event_region', 'stage']] = ['Asia-Pacific', 'Playoffs']
    events_df.loc[events_df.event == 'Major Tiebreaker', ['event_region', 'stage']] = ['North America',
                                                                                       'Main Event Tiebreaker']
    events_df.loc[cond_closed_tb, ['stage', 'event_tier']] = ['Qualifier Tiebreaker', 'Qualifier']
    events_df.loc[cond_main_tb, ['stage', 'event_tier']] = ['Main Event Tiebreaker', 'Qualifier']
    events_df.loc[events_df.stage_is_qualifier, 'event_tier'] = 'Qualifier'  # Fix 'event_tier' for qualifiers

    # Create 'event_phase' from 'stage_is_qualifier' and patch specific rows (event_phase and stage)
    events_df['event_phase'] = np.where(events_df.stage_is_qualifier.isin([True]), events_df.stage, 'Main Event')
    events_df.loc[events_df.event == 'APAC Qualifier', 'event_phase'] = 'Major Qualifier'

    # Fix 'stage'
    cond_tb = (events_df.event_phase == 'Main Event Tiebreaker') | (events_df.event_phase == 'Qualifier Tiebreaker')
    cond_spring = (events_df.event_split == 'Spring') & (events_df.event_phase == 'Main Event')

    events_df.loc[(events_df.event_phase == 'Closed Qualifier') |
                  (events_df.event_phase == 'Invitational Qualifier'), 'stage'] = 'Swiss Stage'

    events_df.loc[cond_tb | cond_spring, 'stage'] = 'Playoffs'

    # Convert date columns to datetime
    date_cols = ['event_start_date', 'event_end_date', 'stage_start_date', 'stage_end_date']
    events_df.loc[:, date_cols] = events_df.loc[:, date_cols].apply(pd.to_datetime)

    # Change event_split to categorical columns for sorting
    events_df.event_split = pd.Categorical(events_df.event_split, ['Fall', 'Winter', 'Spring', 'Summer'])

    events_df.drop('event_name', axis=1, inplace=True)  # Drop irrelevant columns 'event_name'
    events_df = events_df.sort_values(['event_split', 'event_start_date',
                                       'stage_start_date']).reset_index(drop=True)  # Sort columns

    # Filter events that have not yet taken place
    events_df = events_df.loc[events_df.stage_end_date < TODAY]

    if csv_export:
        events_df.to_csv('../../data/retrieved/events.csv', encoding='utf8', index=False)

    return events_df


def parse_matches(events_dataframe: pd.DataFrame, matches_export: bool = False, workers: int = 8):
    """Parse events matches
    :param events_dataframe: event pandas dataframe with 'event_id' (column containing octane.gg event identifiers)
    :param matches_export: to export matches dataframes as .csv files or not
    :param workers: CPU used for multiprocessing
    :return: games dataframe and matches dataframes (by players and by teams)
    """
    # Data collection by events identifier
    event_ids_list = events_dataframe['event_id'].unique()
    matches_p_map = p_tqdm.p_map(get_event_matches, event_ids_list, num_cpus=workers, desc='Matches requests')
    matches_list = list(itertools.chain(*matches_p_map))

    # Flat 'matches_list' and drop irrelevant columns
    first_drop = ['event_slug', 'event_name', 'event_region', 'event_mode', 'event_tier', 'event_image',
                  'event_groups', 'stage_name', 'stage_qualifier', 'format_type', 'stage_lan', 'blue_team_team_image',
                  'orange_team_team_image']

    matches_df = pd.json_normalize(matches_list, sep='_').drop(first_drop, axis=1).dropna(axis=1, how='all')
    matches_df.loc[:, 'date'] = matches_df.loc[:, 'date'].apply(pd.to_datetime)  # Convert date columns to datetime
    matches_df.format_length = 'best-of-' + matches_df.format_length.astype(str)  # Redefine format column

    # Filter matches that have not yet taken place
    matches_df = matches_df.loc[matches_df.date < TODAY]

    # Filter columns
    team_blue_cols = [col for col in matches_df.columns if 'blue_team_' in col] + ['blue_score', 'blue_winner']
    team_oran_cols = [col for col in matches_df.columns if 'orange_team_' in col] + ['orange_score', 'orange_winner']
    team_neut_cols = [col for col in matches_df.columns if
                      col not in team_blue_cols + team_oran_cols + ['blue_players', 'orange_players', 'games']]

    # Games temporary dataframes
    games_tmp_df = pd.json_normalize(matches_df.loc[:, ['_id', 'games']].explode('games').to_dict('records'), sep='_')
    games_tmp_df = games_tmp_df.loc[:, ['_id', 'games__id']].rename(columns={'_id': 'match_id', 'games__id': 'game_id'})

    # Matches dataframes
    teams_blue_df = format_teams_df(ref_df=matches_df, neut_cols=team_neut_cols, side_cols=team_blue_cols, color='blue')
    teams_oran_df = format_teams_df(ref_df=matches_df, neut_cols=team_neut_cols, side_cols=team_oran_cols,
                                    color='orange')

    matches_teams_df = pd.concat([teams_blue_df, teams_oran_df])  # Concat bot side
    matches_teams_df.columns = matches_teams_df.columns.str.replace('stats_', '')  # Remove stats. column prefix
    matches_teams_df.team_name = matches_teams_df.team_name.str.upper()  # Team name to upper case
    matches_teams_df.drop('team_relevant', axis=1, inplace=True)

    # Format columns names in lower case with underscore as separator and rename some of them
    matches_teams_df.columns = matches_teams_df.columns.to_series().apply(format_string_underscore)
    matches_teams_df.rename(columns=true_cols['renaming']['matches'], inplace=True)

    # Players dataframes
    blue_sup_col = ['blue_team_team__id', 'blue_team_team_region', 'blue_score']
    oran_sup_col = ['orange_team_team__id', 'orange_team_team_region', 'orange_score']

    matches_blue_df = format_players_df(ref_df=matches_df, neut_cols=team_neut_cols, side_cols=blue_sup_col,
                                        color='blue')

    matches_oran_df = format_players_df(ref_df=matches_df, neut_cols=team_neut_cols, side_cols=oran_sup_col,
                                        color='orange')

    matches_players_df = pd.concat([matches_blue_df, matches_oran_df])  # Concat bot side
    matches_players_df.columns = matches_players_df.columns.str.replace('stats_', '')  # Remove stats. column prefix

    # Drop irrelevant columns
    to_drop = ['player_team__id', 'player_team_slug', 'player_team_name', 'player_team_region', 'player_team_image',
               'player_accounts', 'player_team_relevant', 'player_name', 'player_relevant']
    matches_players_df.drop(to_drop, axis=1, inplace=True)

    matches_players_df.columns = matches_players_df.columns.str.replace('player_team_', 'team_')  # Change team prefix

    # Format columns names in lower case with underscore as separator and rename some of them
    matches_players_df.columns = matches_players_df.columns.to_series().apply(format_string_underscore)
    matches_players_df.rename(columns=true_cols['renaming']['matches'], inplace=True)

    # Merge matches_teams_df with some elements from events dataframe
    events_reduced = events_dataframe.loc[:, ['event_id', 'event_start_date', 'stage_start_date', 'event_split',
                                              'event', 'event_phase', 'stage', 'stage_step']]

    matches_teams_df = events_reduced.merge(matches_teams_df)
    matches_players_df = events_reduced.merge(matches_players_df)

    # Add matches round to dataframes
    matches_teams_df = add_rounds(matches_df=matches_teams_df)
    matches_players_df = add_rounds(matches_df=matches_players_df)

    # Reorder games_tmp_df
    games_tmp_df = matches_teams_df.loc[:, ['event_id', 'match_id']] \
        .merge(games_tmp_df) \
        .drop_duplicates() \
        .dropna(subset=['game_id'])

    # Fill NaN
    fill_na = {'winner': False, 'reverse_sweep': False, 'reverse_sweep_attempt': False, 'score': 0}
    matches_teams_df.fillna(value=fill_na, inplace=True)
    matches_players_df.fillna(value=fill_na, inplace=True)
    matches_players_df = missing_player_country(dataframe=matches_players_df)

    if matches_export:  # Export dataframes as CSV files
        matches_teams_df.to_csv('../../data/retrieved/matches_by_teams.csv', encoding='utf8', index=False)
        matches_players_df.to_csv('../../data/retrieved/matches_by_players.csv', encoding='utf8', index=False)

    return games_tmp_df, matches_teams_df, matches_players_df


def parse_games(init_games_df: pd.DataFrame, games_export: bool = False, workers: int = 8):
    """Parse events matches
    :param init_games_df: initial pandas dataframe with 'game_id' (column containing octane.gg games identifiers)
    :param games_export: to export games dataframes as .csv files or not
    :param workers: CPU used for multiprocessing
    :return: games dataframes by teams and by players
    """
    # Retrieving games details
    games_list = p_tqdm.p_map(get_game_details, init_games_df.game_id, num_cpus=workers, desc='Games requests')

    # Flat the list of dict. as pandas dataframe
    games_df = pd.json_normalize(games_list, sep='_').dropna(axis=1, how='all')
    games_df = missing_map_name_id(dataframe=games_df)  # Fill missing map_name
    games_df.loc[:, 'date'] = games_df.loc[:, 'date'].apply(pd.to_datetime)  # Date column to datetime
    games_df.number = games_df.number.astype('int64')  # Game number to integer

    # Filter match columns (irrelevant here)
    match_cols = [col for col in games_df.columns if 'match_' in col] + ['blue_matchWinner', 'orange_matchWinner']

    # Merge with init_games_df to have event_id and stage_step columns, drop match columns
    games_df = init_games_df.merge(games_df.drop(match_cols, axis=1).rename(columns={'_id': 'game_id'}))

    # Get stats columns by teams (blue / orange) and neutral columns
    team_blue_cols = [col for col in games_df.columns if 'blue_team_' in col] + ['blue_winner']
    team_oran_cols = [col for col in games_df.columns if 'orange_team_' in col] + ['orange_winner']
    team_neut_cols = [col for col in games_df.columns if
                      col not in team_oran_cols + team_blue_cols + ['blue_players', 'orange_players']]

    # Games dataframes
    teams_blue_df = format_teams_df(ref_df=games_df, neut_cols=team_neut_cols, side_cols=team_blue_cols, color='blue')
    teams_oran_df = format_teams_df(ref_df=games_df, neut_cols=team_neut_cols, side_cols=team_oran_cols, color='orange')

    # Concat both sides
    games_teams = pd.concat([teams_blue_df, teams_oran_df]) \
        .drop(['team_image', 'team_relevant', 'flipBallchasing'], axis=1)

    games_teams.columns = games_teams.columns.str.replace('stats_', '')  # Delete prefix for stats columns

    # Format columns names in lower case with underscore as separator and rename some of them
    games_teams.columns = games_teams.columns.to_series().apply(format_string_underscore)
    games_teams.rename(true_cols['renaming']['games'], axis=1, inplace=True)
    games_teams.team_name = games_teams.team_name.str.upper()  # Team name to upper case

    blue_sup_col = ['blue_team_team__id', 'blue_team_team_region']
    oran_sup_col = ['orange_team_team__id', 'orange_team_team_region']

    games_blue_df = format_players_df(ref_df=games_df, neut_cols=team_neut_cols, side_cols=blue_sup_col, color='blue')
    games_oran_df = format_players_df(ref_df=games_df, neut_cols=team_neut_cols, side_cols=oran_sup_col, color='orange')

    games_players = pd.concat([games_blue_df, games_oran_df])  # Concat bot side
    games_players.columns = games_players.columns.str.replace('stats_', '')  # Remove stats. column prefix

    # Drop irrelevant columns
    to_drop = ['player_team__id', 'player_team_slug', 'player_team_name', 'player_team_region', 'player_team_image',
               'player_accounts', 'player_team_relevant', 'player_name', 'player_relevant', 'flipBallchasing']

    games_players.drop(to_drop, axis=1, inplace=True)
    games_players.columns = games_players.columns.str.replace('player_team_', 'team_')  # Change team prefix

    # Format columns names in lower case with underscore as separator and rename some of them
    games_players.columns = games_players.columns.to_series().apply(format_string_underscore)
    games_players.rename(columns=true_cols['renaming']['games'], inplace=True)

    # Fill NaN
    fill_na = {'winner': False, 'overtime': False, 'advanced_mvp': False}
    games_teams.fillna(value=fill_na, inplace=True)
    games_players.fillna(value=fill_na, inplace=True)
    games_players = missing_player_country(dataframe=games_players)

    if games_export:  # Export dataframes as CSV files
        games_teams.to_csv('../../data/retrieved/games_by_teams.csv', encoding='utf8', index=False)
        games_players.to_csv('../../data/retrieved/games_by_players.csv', encoding='utf8', index=False)

    return games_teams, games_players


def parse_players(game_df: pd.DataFrame, workers: int = 8):
    """Parse events matches
    :param game_df: pandas dataframe with 'player_id' (column containing octane.gg players identifiers)
    :param workers: CPU used for multiprocessing
    :return players_db: games dataframes by teams and by players.
    """
    # Retrieving players details / Set columns to drop and rename
    players_list = p_tqdm.p_map(get_player, game_df.player_id.unique(), num_cpus=workers, desc='Players requests')
    to_drop = ['slug', 'country', 'team', 'relevant', 'substitute', 'coach']
    to_rename = {'_id': 'player_id', 'tag': 'player_tag', 'accounts_platform': 'platform',
                 'accounts_id': 'platform_id', 'name': 'player_name'}

    # Convert list to dataframe
    players_db = pd.json_normalize(pd.DataFrame(players_list).drop(to_drop, axis=1).explode('accounts')
                                   .to_dict('records'), sep='_').rename(columns=to_rename)

    return players_db


def reduce_dataframes(event_df: pd.DataFrame, match_team: pd.DataFrame, match_player: pd.DataFrame,
                      game_team: pd.DataFrame, game_player: pd.DataFrame, export_data: bool = True,
                      export_coverage: bool = True):
    """Reduce, sort and save existing dataframes / create and save a main dataframe with every single match information
    :param event_df: octane.gg events dataframes
    :param match_team: octane.gg matches by teams dataframes
    :param match_player: octane.gg matches by players dataframes
    :param game_team: octane.gg games by teams dataframes
    :param game_player: octane.gg games by players dataframes
    :param export_data: to export dataframes as .csv files or not
    :param export_coverage: to export event coverage as .txt file
    :return: main_df, match_team, match_player, game_team, game_player reduced and sorted.
    """

    def fix_team_region(dataframe: pd.DataFrame):
        """Fix team_region
        :param dataframe: team/players dataframe
        :return dataframe: fixed dataframe.
        """
        dataframe.loc[(dataframe.event_region != 'World') &
                      (dataframe.event_region != 'Asia-Pacific'), 'team_region'] = dataframe.event_region

        apac_qualifiers = true_cols['apac_qualifiers']
        missing_region = missing_value['teams_region']

        for split, teams_list in apac_qualifiers.items():
            for team in teams_list:
                dataframe.loc[(dataframe.team_id == team['team_id']) &
                              (dataframe.event_split == split), 'team_region'] = team['team_region']

        for team in missing_region:
            dataframe.loc[dataframe.team_id == team['team_id'], 'team_region'] = team['team_region']

        world_events = {'EU': 'Europe', 'NA': 'North America', 'OCE': 'Oceania', 'SAM': 'South America',
                        'ME': 'Middle East & North Africa', 'AF': 'Sub-Saharan Africa', 'ASIA': 'Asia-Pacific North'}

        dataframe.team_region.replace(world_events, inplace=True)

        return dataframe

    matches_reduced = match_team.loc[:, ['event_id', 'stage_step', 'match_id', 'match_slug', 'match_number',
                                         'match_round', 'match_date', 'match_format', 'reverse_sweep_attempt',
                                         'reverse_sweep']].drop_duplicates()

    games_reduced = game_team.loc[:, ['event_id', 'match_id', 'game_id', 'game_number', 'game_date', 'ballchasing_id',
                                      'game_duration', 'map_id', 'map_name', 'overtime']] \
        .drop_duplicates()

    main_df = event_df.merge(matches_reduced).merge(games_reduced, how='outer')

    # Change event_split to categorical columns for sorting
    split_order = ['Fall', 'Winter', 'Spring', 'Summer']
    event_order = ['Regional 1', 'Regional 2', 'Regional 3', 'APAC Qualifier', 'Major Tiebreaker', 'Major',
                   'World Championship']
    phase_order = ['Invitational Qualifier', 'Qualifier Tiebreaker', 'Closed Qualifier', 'Main Event Tiebreaker',
                   'Main Event', 'Major Qualifier']

    main_df.event_split = pd.Categorical(main_df.event_split, split_order)
    main_df.event = pd.Categorical(main_df.event, event_order)
    main_df.event_phase = pd.Categorical(main_df.event_phase, phase_order)

    # Sort Main Dataframe
    sorting_order = ['event_split', 'event', 'event_start_date', 'event_phase', 'stage_start_date',
                     'stage_step', 'match_number', 'game_number']

    main_df = main_df.sort_values(sorting_order).reset_index(drop=True)

    # Merge with Main Dataframe to sort other dataframes
    match_team, match_player = main_df.merge(match_team), main_df.merge(match_player)
    game_team, game_player = main_df.merge(game_team), main_df.merge(game_player)

    # Fix regions
    match_team, match_player = fix_team_region(dataframe=match_team), fix_team_region(dataframe=match_player)
    game_team, game_player = fix_team_region(dataframe=game_team), fix_team_region(dataframe=game_player)

    # Drop columns (and duplicated rows) from stats dataframe to reduced their size to minimum
    to_drop = ['event_id', 'event_slug', 'event_start_date', 'event_end_date', 'event_region', 'event_tier',
               'prize_money', 'stage_step', 'stage', 'stage_start_date', 'stage_end_date', 'liquipedia_link',
               'stage_is_lan', 'location_venue', 'location_city', 'location_country', 'stage_is_qualifier',
               'event', 'event_split', 'event_phase', 'match_number', 'match_round', 'match_date', 'match_format',
               'game_number', 'game_date', 'ballchasing_id', 'match_slug', 'reverse_sweep_attempt', 'reverse_sweep',
               'game_duration', 'map_id', 'map_name', 'overtime']

    match_team = match_team.drop(to_drop + ['game_id'], axis=1).drop_duplicates()
    match_player = match_player.drop(to_drop + ['game_id'], axis=1).drop_duplicates()
    game_team = game_team.drop(to_drop + ['match_id'], axis=1).drop_duplicates()
    game_player = game_player.drop(to_drop + ['match_id'], axis=1).drop_duplicates()

    # Order main_df columns
    main_df = main_df.reindex(columns=true_cols['ordering']['main'])

    # Reset indexes
    main_df.reset_index(drop=True, inplace=True)
    match_team.reset_index(drop=True, inplace=True)
    match_player.reset_index(drop=True, inplace=True)
    game_team.reset_index(drop=True, inplace=True)
    game_player.reset_index(drop=True, inplace=True)

    # Set URLs
    main_df.event_slug = 'https://octane.gg/events/' + main_df.event_slug.map(str)
    main_df.match_slug = 'https://octane.gg/matches/' + main_df.match_slug.map(str)
    game_team.team_slug = 'https://octane.gg/teams/' + game_team.team_slug.map(str)
    match_team.team_slug = 'https://octane.gg/teams/' + match_team.team_slug.map(str)

    if export_data:  # Export dataframes as CSV files
        main_df.to_csv('../../data/retrieved/main.csv', encoding='utf8', index=False)
        match_team.to_csv('../../data/retrieved/matches_by_teams.csv', encoding='utf8', index=False)
        match_player.to_csv('../../data/retrieved/matches_by_players.csv', encoding='utf8', index=False)
        game_team.to_csv('../../data/retrieved/games_by_teams.csv', encoding='utf8', index=False)
        game_player.to_csv('../../data/retrieved/games_by_players.csv', encoding='utf8', index=False)

    if export_coverage:  # Export events covered by the datasets as CSV files
        coverage = main_df.loc[:, ['event_id', 'event_split', 'event', 'event_region', 'event_phase']].drop_duplicates()
        coverage.to_csv('../../data/public/data_coverage.csv', index=False, encoding='utf8')

    return main_df, match_team, match_player, game_team, game_player


def collect_ballchasing(game_df: pd.DataFrame, token: str = my_token, workers: int = 8, n_chunk: int = 10):
    """Retrieve ballchasing replay details
    :param game_df: pandas dataframe with 'player_id' (column containing octane.gg players identifiers)
    :param token: ballchasing.com API token
    :param workers: CPU used for multiprocessing
    :param n_chunk: subdivisions number for iterations on IDs
    :return replay_list: list of replays with supplementary information.
    """
    if replays_tmp:  # Check if some information are already retrieved
        already_done = {replay['id'] for replay in replays_tmp if 'id' in replay.keys()}
        to_treat = [replay_id for replay_id in game_df.ballchasing_id.unique() if replay_id not in already_done]
        replay_list = replays_tmp

    else:
        to_treat = game_df.ballchasing_id.unique().tolist()
        replay_list = []

    if len(to_treat) > n_chunk * 2:  # Check if multiprocessing is necessary (a lot of element to treat)
        split_list = np.array_split(to_treat, n_chunk)

        for idx, chunk in enumerate(split_list):
            replay_tmp = p_tqdm.p_map(get_ballchasing, chunk, [token] * len(chunk), num_cpus=workers,
                                      desc=f'ballchasing.com requests chunk {idx + 1}/{n_chunk}')
            replay_list += [replay for replay in replay_tmp if 'id' in replay.keys()]  # Cover Error 429

            with open('../../data/retrieved/replays_tmp.json', 'w', encoding='utf-8') as replay_list_file:
                json.dump(replay_list, replay_list_file, indent=1)  # Saving intermediate results

    elif to_treat:
        for ballchasing_id in tqdm.tqdm(to_treat, desc='ballchasing.com requests'):
            replay_list += [get_ballchasing(ballchasing_id)]

            with open('../../data/retrieved/replays_tmp.json', 'w', encoding='utf-8') as replay_list_file:
                json.dump(replay_list, replay_list_file, indent=1)  # Saving intermediate results

    else:
        print('All replays already retrieved!')

    return replay_list


def parse_ballchasing(ballchasing_list: list):
    """Create a dataframe from information collected with ballchasing.com API
    :param ballchasing_list: ballchasing.com list with replay ID, camera settings, ...
    :return: a clean dataframe.
    """
    # Flat the list of dictionaries into a dataframe
    ballchasing_df = pd.json_normalize(pd.DataFrame(ballchasing_list)
                                       .explode('players')
                                       .to_dict('records'), sep='_').drop('players', axis=1)

    ballchasing_df.columns = ballchasing_df.columns.str.replace('players_', '')  # Remove prefix
    ballchasing_df.rename(columns=true_cols['renaming']['ballchasing'], inplace=True)  # Rename columns
    return ballchasing_df


def complete_player_df(main: pd.DataFrame, match_player: pd.DataFrame, game_player: pd.DataFrame,
                       workers_octanegg: int = 8, workers_ballchasing: int = 8, export_data: bool = True):
    """Add platform information, settings and cars used to game by players dataframe
    :param main: main dataframe with events, matches and games information
    :param match_player: match by players dataframe
    :param game_player: game by players dataframe
    :param workers_octanegg: CPU resources used for multiprocessing tasks with octane.gg API
    :param workers_ballchasing: CPU resources used for multiprocessing tasks with ballchasing.com API
    :param export_data: to export dataframes as .csv files or not
    :return: players DB, matches & games dataframes enhanced with new information.
    """

    def fix_mvp(game_player_df: pd.DataFrame):
        """Fix missing MVP attribution
        :param game_player_df: game by players dataframe
        :return: fixed dataframe.
        """
        game_player_reduced = game_player_df.loc[game_player_df.winner.isin([True]),
                                                 ['game_id', 'player_id', 'core_score', 'advanced_mvp']] \
            .sort_values(['game_id', 'core_score'], ascending=False) \
            .drop_duplicates(subset=['game_id'])

        missing_mvp = game_player_reduced.loc[game_player_reduced.advanced_mvp.isin([False])] \
            .drop(['core_score', 'advanced_mvp'], axis=1) \
            .to_dict('records')

        for game in missing_mvp:
            game_player_df.loc[(game_player_df.game_id == game['game_id']) &
                               (game_player_df.player_id == game['player_id']), 'advanced_mvp'] = True

        return game_player_df

    # Save row order
    match_player['row_order'] = match_player.index
    game_player['row_order'] = game_player.index

    # Reduced input dataframes and merge
    main_reduced = main.loc[:, ['game_id', 'ballchasing_id']]
    game_reduced = game_player.loc[:, ['game_id', 'player_id']]
    reduced_df = main_reduced.merge(game_reduced).dropna(subset=['ballchasing_id'])

    # Retrieve missing data
    players = parse_players(game_df=reduced_df, workers=workers_octanegg)
    ballchasing_list = collect_ballchasing(game_df=reduced_df, workers=workers_ballchasing)
    ballchasing_df = parse_ballchasing(ballchasing_list=ballchasing_list)

    # Merge successively to complete input game_player (drop redundant ballchasing_id) & match_player
    players_augmented = game_player.merge(game_reduced.merge(players).merge(main_reduced).merge(ballchasing_df),
                                          how='outer').drop('ballchasing_id', axis=1)

    match_augmented = players.loc[:, ['player_id', 'player_name']].drop_duplicates().merge(match_player)

    players_augmented = missing_car_name(dataframe=players_augmented)  # Fix missing car name
    players_augmented = fix_mvp(game_player_df=players_augmented)  # Fix missing MVP attribution

    # Sort rows, order columns & reset indexes
    match_augmented.sort_values('row_order', inplace=True)
    players_augmented.sort_values('row_order', inplace=True)
    match_augmented = match_augmented.reindex(columns=true_cols['ordering']['matches_by_players'])
    players_augmented = players_augmented.reindex(columns=true_cols['ordering']['games_by_players'])
    match_augmented.reset_index(drop=True, inplace=True)
    players_augmented.reset_index(drop=True, inplace=True)

    # Set players database
    players_db = players_augmented.loc[:, ['player_id', 'player_slug', 'player_tag', 'player_name', 'player_country']] \
        .sort_values(['player_tag', 'player_name'], key=lambda col: col.str.replace('.', '', regex=True).str.lower()) \
        .drop_duplicates(subset=['player_id']) \
        .reset_index(drop=True)

    players_db.player_slug = 'https://octane.gg/players/' + players_db.player_slug.map(str)

    # Drop redundant columns
    match_augmented.drop(['player_slug', 'player_name', 'player_country'], inplace=True, axis=1)
    players_augmented.drop(['player_slug', 'player_name', 'player_country'], inplace=True, axis=1)

    if export_data:
        match_augmented.to_csv('../../data/retrieved/matches_by_players.csv', encoding='utf8', index=False)
        players_augmented.to_csv('../../data/retrieved/games_by_players.csv', encoding='utf8', index=False)
        players_db.to_csv('../../data/retrieved/players_db.csv', encoding='utf8', index=False)

    return match_augmented, players_augmented, players_db


"MAIN"

if __name__ == '__main__':
    # Event Dataframe
    MISSING_EVENTS = ['620bf77fda9d7ca1c7ba8719', '62a054f7da9d7ca1c7bb227e']  # Winter / Spring Major - APAC Qualifier
    EVENTS_DF = parse_events(events_group=rlcs2122, event_id_list=MISSING_EVENTS)

    # Games temp. Dataframe & Matches Dataframes / Set less workers if your CPU doesn't have 15 cores
    G_TMP, MATCHES_TEAMS_TMP, MATCHES_PLAYERS_TMP = parse_matches(events_dataframe=EVENTS_DF, workers=15)

    # Games dataframes / Set less workers if your CPU doesn't have 15 cores
    GAMES_TEAMS_TMP, GAMES_PLAYERS_TMP = parse_games(init_games_df=G_TMP, workers=15)

    # Reduce and save results
    MAIN_DF, MATCH_TEAM, MATCH_PLAYER, GAME_TEAM, GAME_PLAYER = reduce_dataframes(event_df=EVENTS_DF,
                                                                                  match_team=MATCHES_TEAMS_TMP,
                                                                                  match_player=MATCHES_PLAYERS_TMP,
                                                                                  game_team=GAMES_TEAMS_TMP,
                                                                                  game_player=GAMES_PLAYERS_TMP)

    # Add missing information (platform identifiers, settings and cars)
    MATCH_PLAYER, GAME_PLAYER, PLAYERS_DB = complete_player_df(main=MAIN_DF,
                                                               match_player=MATCH_PLAYER,
                                                               game_player=GAME_PLAYER,
                                                               workers_octanegg=15)
