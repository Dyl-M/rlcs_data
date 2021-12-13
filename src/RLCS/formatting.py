# -*- coding: utf-8 -*-

import json
import pandas as pd

"""File Informations

@file_name: formatting.py
@author: Dylan "dyl-m" Monfret
"""

"PREPARATORY ELEMENTS"

with open('../../data/retrieved/raw.json', 'r', encoding='utf8') as json_file:
    dataset_json = json.load(json_file)

with open('../../data/public/alias.json', 'r', encoding='utf8') as alias_file:
    alias = json.load(alias_file)

with open('../../data/public/missing_value.json', 'r', encoding='utf8') as missing_value_file:
    missing_value = json.load(missing_value_file)

"FUNCTIONS"


def map_missing_team_name(dataframe):
    """

    :param dataframe:
    :return:
    """
    for best_of in missing_value["team_names"]:
        blue_name = best_of['blue_name']
        orange_name = best_of['orange_name']

        for match in best_of['ballchasing_id']:
            dataframe.loc[dataframe['ballchasing_id'] == match, 'orange_name'] = orange_name
            dataframe.loc[dataframe['ballchasing_id'] == match, 'blue_name'] = blue_name

    return dataframe


def map_missing_car_name(dataframe):
    """

    :param dataframe:
    :return:
    """
    for car in missing_value['car_names']:
        car_name = car['car_name']
        car_id = car['car_id']
        dataframe.loc[dataframe['p_car_id'] == car_id, 'p_car_name'] = car_name

    return dataframe


"MAIN"

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)

    # Flat entire JSON

    main_dataframe = pd.json_normalize(dataset_json, sep='_')
    main_dataframe = main_dataframe.rename(
        columns={col: col.replace('details_', '').replace('stats_', '') for col in list(main_dataframe.columns)})

    # Fill empty team name field

    main_dataframe = map_missing_team_name(main_dataframe)
    # print(main_dataframe[main_dataframe['orange_name'].isna() | main_dataframe['blue_name'].isna()])

    # Replacing team alias

    main_dataframe['orange_name'] = main_dataframe['orange_name'].replace(alias)
    main_dataframe['blue_name'] = main_dataframe['blue_name'].replace(alias)
    # print(set(main_dataframe['orange_name'].unique().tolist() + main_dataframe['blue_name'].unique().tolist()))

    # Flat ballchasing.com groups

    groups_dataframe = main_dataframe[['ballchasing_id', 'groups']].explode('groups').to_dict('records')
    groups_dataframe = pd.json_normalize(groups_dataframe, sep='_')

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
    by_players_dataframe['p_positioning_goals_against_while_last_defender'] = by_players_dataframe[
        'p_positioning_goals_against_while_last_defender'].fillna(0)
    # by_players_dataframe['p_mvp'] = by_players_dataframe['p_mvp'].fillna(False)

    # Filtering / Formatting / Timezone normalization / Best-of ID generation

    main_dataframe = main_dataframe.drop(['orange_players', 'blue_players', 'groups'], axis=1)
    main_dataframe['created'] = pd.to_datetime(main_dataframe['created'], utc=True)
    main_dataframe['date'] = pd.to_datetime(main_dataframe['date'], utc=True)

    general_dataframe = pd \
        .concat([main_dataframe.iloc[:, :30], main_dataframe.iloc[:, -3:]], axis=1) \
        .sort_values(by='created')

    bo_id_df = general_dataframe[['region', 'split', 'event', 'phase',
                                  'stage', 'round', 'match']].drop_duplicates().reset_index(drop=True)

    bo_id_df['bo_id'] = bo_id_df.index  # .apply(str)
    bo_id_df['bo_id'] = 'B0_' + bo_id_df['bo_id'].apply(str).str.zfill(5)

    general_dataframe = pd.merge(bo_id_df,
                                 general_dataframe,
                                 on=['region', 'split', 'event', 'phase', 'stage', 'round', 'match'])

    general_df_cols_ordered = ['ballchasing_id', 'bo_id', 'link', 'region', 'split', 'event', 'phase', 'stage', 'round',
                               'match', 'created', 'uploader_steam_id', 'uploader_name', 'uploader_profile_url',
                               'uploader_avatar', 'status', 'rocket_league_id', 'match_guid', 'title', 'recorder',
                               'map_code', 'map_name', 'match_type', 'team_size', 'playlist_id', 'playlist_name',
                               'duration', 'overtime', 'overtime_seconds', 'season', 'season_type', 'date',
                               'date_has_timezone', 'visibility']

    general_dataframe = general_dataframe.reindex(columns=general_df_cols_ordered)

    # Team stats only

    by_team_dataframe = main_dataframe.drop(columns=general_df_cols_ordered[2:])
    or_col = [col for col in list(by_team_dataframe.columns) if 'orange' in col or 'ballchasing' in col]
    bl_col = [col for col in list(by_team_dataframe.columns) if 'blue' in col or 'ballchasing' in col]
    or_side = by_team_dataframe[or_col].rename(columns={col: col.replace('orange_', '') for col in or_col})
    bl_side = by_team_dataframe[bl_col].rename(columns={col: col.replace('blue_', '') for col in bl_col})

    by_team_dataframe = bl_side.append(or_side)

    # Exports

    # by_team_dataframe.to_csv('../../data/retrieved/by_teams.csv', encoding='utf8', index=False)
    # by_players_dataframe.to_csv('../../data/retrieved/by_players.csv', encoding='utf8', index=False)
    # groups_dataframe.to_csv('../../data/retrieved/groups.csv', encoding='utf8', index=False)
    # general_dataframe.to_csv('../../data/retrieved/general.csv', encoding='utf8', index=False)
