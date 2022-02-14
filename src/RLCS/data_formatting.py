# -*- coding: utf-8 -*-

import json
import pandas as pd

"""File Informations

@file_name: data_formatting.py
@author: Dylan "dyl-m" Monfret

Things to verify before formatting / export:
1- Missing values (teams names and cars names) to fix (missing_values.json).
2- New team alias (alias.json) to fix too.
"""

"IMPORTS"

with open('../../data/retrieved/raw.json', 'r', encoding='utf8') as json_file:
    dataset_json = json.load(json_file)

with open('../../data/public/alias.json', 'r', encoding='utf8') as alias_file:
    alias = json.load(alias_file)

with open('../../data/public/missing_value.json', 'r', encoding='utf8') as missing_value_file:
    missing_value = json.load(missing_value_file)

"FUNCTIONS"


def first_sorting(dataframe: pd.DataFrame):
    """Fill team name field the right way
    :param dataframe: main / original pandas dataframe
    :return: sorted dataframe.
    """
    dataframe = dataframe \
        .replace('Fall', 'Split 1 - Fall') \
        .replace('Winter', 'Split 2 - Winter') \
        .replace('World', '0 - World') \
        .replace('Europe', '1 - Europe') \
        .replace('North America', '2 - North America') \
        .replace('Oceania', '3 - Oceania') \
        .replace('South America', '4 - South America') \
        .replace('Middle East & North Africa', '5 - Middle East & North Africa') \
        .replace('Asia-Pacific North', '6 - Asia-Pacific North') \
        .replace('Asia-Pacific South', '7 - Asia-Pacific South') \
        .replace('Sub-Saharan Africa', '8 - Sub-Saharan Africa') \
        .replace('Major', '0 Major') \
        .replace('Playoffs', '1 Playoffs') \
        .replace('Swiss', '0 Swiss') \
        .replace('Groups', '0 Groups') \
        .replace('Round 1', '0 Round 1') \
        .replace('Round 2', '0 Round 2') \
        .replace('Round 3', '0 Round 3') \
        .replace('Round 4', '0 Round 4') \
        .replace('Round 5', '0 Round 5') \
        .replace('Tiebreaker Round', '0 Tiebreaker Round') \
        .replace('Lower Round 1', '1 Lower Round 1') \
        .replace('Lower Round 2', '2 Lower Round 2') \
        .replace('Quarterfinal', '2 Quarterfinal') \
        .replace('Lower Quarterfinal', '3 Lower Quarterfinal') \
        .replace('Semifinal', '4 Semifinal') \
        .replace('Upper Semifinal', '4 Upper Semifinal') \
        .replace('Lower Semifinal', '5 Lower Semifinal') \
        .replace('Upper Final', '6 Upper Final') \
        .replace('Lower Final', '7 Lower Final') \
        .replace('Grand Final', '8 Grand Final') \
        .replace('Final', '8 Final') \
        .sort_values(['split', 'region', 'event', 'stage', 'round', 'match', 'created']) \
        .replace('Split 1 - Fall', 'Fall') \
        .replace('Split 2 - Winter', 'Winter') \
        .replace('0 - World', 'World') \
        .replace('1 - Europe', 'Europe') \
        .replace('2 - North America', 'North America') \
        .replace('3 - Oceania', 'Oceania') \
        .replace('4 - South America', 'South America') \
        .replace('5 - Middle East & North Africa', 'Middle East & North Africa') \
        .replace('6 - Asia-Pacific North', 'Asia-Pacific North') \
        .replace('7 - Asia-Pacific South', 'Asia-Pacific South') \
        .replace('8 - Sub-Saharan Africa', 'Sub-Saharan Africa') \
        .replace('0 Major', 'Major') \
        .replace('1 Playoffs', 'Playoffs') \
        .replace('0 Swiss', 'Swiss') \
        .replace('0 Groups', 'Groups') \
        .replace('0 Round 1', 'Round 1') \
        .replace('0 Round 2', 'Round 2') \
        .replace('0 Round 3', 'Round 3') \
        .replace('0 Round 4', 'Round 4') \
        .replace('0 Round 5', 'Round 5') \
        .replace('0 Tiebreaker Round', 'Tiebreaker Round') \
        .replace('1 Lower Round 1', 'Lower Round 1') \
        .replace('2 Lower Round 2', 'Lower Round 2') \
        .replace('2 Quarterfinal', 'Quarterfinal') \
        .replace('3 Lower Quarterfinal', 'Lower Quarterfinal') \
        .replace('4 Semifinal', 'Semifinal') \
        .replace('4 Upper Semifinal', 'Upper Semifinal') \
        .replace('5 Lower Semifinal', 'Lower Semifinal') \
        .replace('6 Upper Final', 'Upper Final') \
        .replace('7 Lower Final', 'Lower Final') \
        .replace('8 Grand Final', 'Grand Final') \
        .replace('8 Final', 'Final') \
        .reset_index(drop=True)

    return dataframe


def map_missing_team_name(dataframe: pd.DataFrame):
    """Fill team name field the right way
    :param dataframe: main / original pandas dataframe
    :return: fixed pandas dataframe.
    """
    for series in missing_value["team_names"]:
        blue_name = series['blue_name']
        orange_name = series['orange_name']

        for match in series['ballchasing_id']:
            dataframe.loc[dataframe['ballchasing_id'] == match, 'orange_name'] = orange_name
            dataframe.loc[dataframe['ballchasing_id'] == match, 'blue_name'] = blue_name

    return dataframe


def map_missing_car_name(dataframe: pd.DataFrame):
    """Add missing car names to dataframe
    :param dataframe: main / original pandas dataframe
    :return: fixed pandas dataframe.
    """
    for car in missing_value['car_names']:
        car_name = car['car_name']
        car_id = car['car_id']
        dataframe.loc[dataframe['p_car_id'] == car_id, 'p_car_name'] = car_name

    return dataframe


def apply_patch_delete(players_df: pd.DataFrame, to_delete_list: list):
    """Delete some selected players from the dataset
    :param players_df: data by players
    :param to_delete_list: list of players to delete from the dataset (list of dictionaries)
    :return: corrected dataframe.
    """
    for player in to_delete_list:
        players_df = players_df.loc[~((players_df.ballchasing_id == player['ballchasing_id']) &
                                      (players_df.p_platform_id == player['p_platform_id']))]

    return players_df


def apply_patch_move_players(players_df: pd.DataFrame, to_move: list):  # TODO: patch teams dataframe
    """Delete some selected players from the dataset
    :param players_df: data by players
    :param to_move: list of players to move to the right side (list of dictionaries)
    :return: corrected dataframe.
    """
    for match in to_move:
        for player in match['players']:
            players_df.loc[(players_df.ballchasing_id == match['ballchasing_id']) &
                           (players_df.p_platform_id == player['p_platform']), 'name'] = player['name']

            players_df.loc[(players_df.ballchasing_id == match['ballchasing_id']) &
                           (players_df.p_platform_id == player['p_platform']), 'color'] = player['color']

    return players_df


def apply_patch(players_df: pd.DataFrame):
    """Patch anomalies in dataframe(s)
    :param players_df: data by players
    :return: patched dataframe(s).
    """
    with open('../../data/public/patch.json', 'r', encoding='utf8') as patch_file:
        patch = json.load(patch_file)

    to_delete = patch['to_delete_players']
    to_move = patch['to_move_players']

    players_df = apply_patch_delete(players_df=players_df, to_delete_list=to_delete)
    players_df = apply_patch_move_players(players_df=players_df, to_move=to_move)

    return players_df


"MAIN"

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)

    # Flat entire JSON
    main_dataframe = pd.json_normalize(dataset_json, sep='_')
    main_dataframe = main_dataframe.rename(
        columns={col: col.replace('details_', '').replace('stats_', '') for col in list(main_dataframe.columns)}) \
        .sort_values(by='created')

    main_dataframe = first_sorting(main_dataframe)  # Apply first sorting
    main_dataframe = map_missing_team_name(main_dataframe)  # Fill empty team name field

    # Replacing team alias
    main_dataframe['orange_name'] = main_dataframe['orange_name'].replace(alias)
    main_dataframe['blue_name'] = main_dataframe['blue_name'].replace(alias)

    # Flat ballchasing.com groups
    groups_dataframe = main_dataframe[['ballchasing_id', 'groups']].explode('groups').to_dict('records')
    groups_dataframe = pd.json_normalize(groups_dataframe, sep='_').drop('groups', axis=1)

    # Flat players stats.
    bl_players_df = main_dataframe[['ballchasing_id', 'blue_color', 'blue_name', 'blue_players']] \
        .explode('blue_players').to_dict('records')

    bl_players_df = pd.json_normalize(bl_players_df, sep='_')

    bl_players_df = bl_players_df \
        .rename(columns={col: col.replace('blue_', '')
                .replace('stats_', '')
                .replace('players_', 'p_')
                .replace('id_id', 'platform_id')
                .replace('id_platform', 'platform') for col in list(bl_players_df.columns)})

    or_players_df = main_dataframe[['ballchasing_id', 'orange_color', 'orange_name', 'orange_players']] \
        .explode('orange_players').to_dict('records')

    or_players_df = pd.json_normalize(or_players_df, sep='_')

    or_players_df = or_players_df \
        .rename(columns={col: col.replace('orange_', '')
                .replace('stats_', '')
                .replace('players_', 'p_')
                .replace('id_id', 'platform_id')
                .replace('id_platform', 'platform') for col in list(or_players_df.columns)})

    # All players
    by_players_dataframe = bl_players_df.append(or_players_df)
    by_players_dataframe = map_missing_car_name(by_players_dataframe)
    by_players_dataframe['p_car_id'] = by_players_dataframe['p_car_id'].apply(str)
    by_players_dataframe['p_platform_id'] = by_players_dataframe['p_platform_id'].apply(str)
    by_players_dataframe['p_positioning_goals_against_while_last_defender'] = by_players_dataframe[
        'p_positioning_goals_against_while_last_defender'].fillna(0)
    by_players_dataframe.dropna(subset=['p_platform'], inplace=True)  # Drops BOT

    # Filtering / Formatting / Timezone normalization / Best-of ID generation
    main_dataframe = main_dataframe.drop(['orange_players', 'blue_players', 'groups'], axis=1)
    main_dataframe['created'] = pd.to_datetime(main_dataframe['created'], utc=True)
    main_dataframe['date'] = pd.to_datetime(main_dataframe['date'], utc=True)

    general_dataframe = pd \
        .concat([main_dataframe.iloc[:, :31], main_dataframe.iloc[:, -5:]], axis=1)

    bo_id_df = general_dataframe[['region', 'split', 'event', 'phase',
                                  'stage', 'round', 'match']].drop_duplicates().reset_index(drop=True)

    bo_id_df['bo_id'] = bo_id_df.index  # .apply(str)
    bo_id_df['bo_id'] = 'B0_' + bo_id_df['bo_id'].apply(str).str.zfill(5)

    general_dataframe = pd.merge(bo_id_df,
                                 general_dataframe,
                                 on=['region', 'split', 'event', 'phase', 'stage', 'round', 'match'])

    general_df_cols_ordered = ['ballchasing_id', 'correction', 'bo_id', 'link', 'region', 'split', 'event', 'phase',
                               'stage', 'round', 'match', 'created', 'uploader_steam_id', 'uploader_name',
                               'uploader_profile_url', 'uploader_avatar', 'status', 'rocket_league_id', 'match_guid',
                               'title', 'recorder', 'map_code', 'map_name', 'match_type', 'team_size', 'playlist_id',
                               'playlist_name', 'duration', 'overtime', 'overtime_seconds', 'season', 'season_type',
                               'date', 'date_has_timezone', 'visibility']

    general_dataframe = general_dataframe.reindex(columns=general_df_cols_ordered)
    general_dataframe['uploader_steam_id'] = general_dataframe['uploader_steam_id'].apply(str)

    # Team stats only
    by_team_dataframe = main_dataframe.drop(columns=general_df_cols_ordered[3:])
    or_col = [col for col in list(by_team_dataframe.columns) if 'orange' in col or 'ballchasing' in col]
    bl_col = [col for col in list(by_team_dataframe.columns) if 'blue' in col or 'ballchasing' in col]
    or_side = by_team_dataframe[or_col].rename(columns={col: col.replace('orange_', '') for col in or_col})
    bl_side = by_team_dataframe[bl_col].rename(columns={col: col.replace('blue_', '') for col in bl_col})

    by_team_dataframe = bl_side.append(or_side)

    # Apply dataset correction
    by_players_dataframe = apply_patch(by_players_dataframe)

    # Exports
    by_team_dataframe.to_csv('../../data/retrieved/by_teams.csv', encoding='utf8', index=False)
    by_players_dataframe.to_csv('../../data/retrieved/by_players.csv', encoding='utf8', index=False)
    groups_dataframe.to_csv('../../data/retrieved/groups.csv', encoding='utf8', index=False)
    general_dataframe.to_csv('../../data/retrieved/general.csv', encoding='utf8', index=False)
