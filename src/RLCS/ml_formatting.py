# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from datetime import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

"""File Information

@file_name: ml_formatting.py
@author: Dylan "dyl-m" Monfret

Treatment to apply on .csv files for ML model conception.
"""

"FUNCTIONS"


def treatment_by_players(ref_date_str, export_players_db: bool = False):
    """Pretreatment pipeline to build a dataframe set for models by players
    :param ref_date_str: A reference date as string, to put a weight on matches based to how old those games are
    :param export_players_db: to export players database (with in-game ID and team) or not
    :return: Dataset formatted for modeling and players name and ID database.
    """
    # Reference date to game datetime
    ref_date = datetime.strptime(ref_date_str, '%Y-%m-%d %H:%M:%S%z')

    # Dataframe imports
    players_df = pd.read_csv('../../data/retrieved/by_players.csv', encoding='utf8', low_memory=False)
    general_df = pd.read_csv('../../data/retrieved/general.csv', encoding='utf8', low_memory=False)

    players_df = players_df \
        .rename(columns={'name': 'team'}) \
        .rename(columns=lambda x: x[2:] if x.startswith('p_') else x)

    # Keep relevant features from each dataset
    general_df = general_df.loc[:, ['ballchasing_id', 'correction', 'region', 'split', 'event', 'phase', 'stage',
                                    'round', 'date', 'duration', 'overtime', 'overtime_seconds']]
    players_df = players_df.drop(['start_time', 'end_time', 'mvp', 'car_id'], axis=1)

    # Merge dataset and drop stats correction replays
    dataframe = general_df.merge(players_df)
    dataframe = dataframe.loc[~dataframe.correction].drop('correction', axis=1)

    # Get matches results
    results = dataframe.loc[:, ['ballchasing_id', 'color', 'core_mvp']] \
        .drop_duplicates() \
        .groupby(['ballchasing_id', 'color'], as_index=False).mean()

    results = results.loc[results.core_mvp > 0].drop('core_mvp', axis=1).rename(columns={'color': 'win'})

    # Merge results with previous data and recode game_winner into 1 if the player won the game, else 0.
    dataframe = dataframe.merge(results)
    dataframe.win = np.where(dataframe.color == dataframe.win, 1, 0)

    # Recode platform ID with "platform + _ + ID"
    dataframe.platform_id = dataframe['platform'] + '_' + dataframe['platform_id'].astype(str)
    dataframe = dataframe.drop(['platform'], axis=1)

    # Changing 'date' to a time delta with the reference date
    dataframe.date = pd.to_datetime(dataframe.date, utc=True)
    dataframe.date = (dataframe.date - ref_date) / np.timedelta64(1, 'D')
    dataframe = dataframe.rename(columns={'date': 'since_ref_date'})

    # Fill overtime seconds
    dataframe.overtime_seconds = dataframe.overtime_seconds.fillna(0)

    # Players name & ID dataset
    players_db = dataframe.loc[:, ['team', 'name', 'platform_id', 'since_ref_date']] \
        .sort_values('since_ref_date', ascending=False) \
        .drop_duplicates(subset=['platform_id'], keep='first') \
        .reset_index(drop=True) \
        .sort_values(['team', 'name']) \
        .drop('since_ref_date', axis=1)

    # Add opponents and teammates as features
    # # Create a reduced dataframe
    df_reduced = dataframe.loc[:, ['ballchasing_id', 'color', 'team', 'platform_id', 'core_score']]

    # # Split blue and orange side and group players ID into list
    bl_side = df_reduced.loc[df_reduced.color == 'blue'] \
        .rename(columns={'platform_id': 'id_list'}) \
        .sort_values(['ballchasing_id', 'core_score'], ascending=False) \
        .groupby(['ballchasing_id', 'color', 'team'])['id_list'] \
        .apply(list) \
        .reset_index()

    or_side = df_reduced.loc[df_reduced.color == 'orange'] \
        .rename(columns={'platform_id': 'id_list'}) \
        .sort_values(['ballchasing_id', 'core_score'], ascending=False) \
        .groupby(['ballchasing_id', 'color', 'team'])['id_list'] \
        .apply(list) \
        .reset_index()

    # # Teammates Dataframe
    # # # Blue side
    bl_teammates_list_v1 = df_reduced.loc[df_reduced.color == 'blue', ['ballchasing_id', 'platform_id']] \
        .merge(bl_side.drop('team', axis=1))

    bl_teammates_ex = bl_teammates_list_v1.explode('id_list').reset_index(drop=True)

    bl_teammates_list_v2 = bl_teammates_ex[bl_teammates_ex.id_list != bl_teammates_ex.platform_id] \
        .groupby(['ballchasing_id', 'platform_id'])['id_list'] \
        .apply(list) \
        .reset_index()

    bl_teammates = pd.concat([bl_teammates_list_v2.loc[:, ['ballchasing_id', 'platform_id']],
                              bl_teammates_list_v2.id_list.apply(pd.Series)], axis=1) \
        .rename(columns={0: 'teammate_1', 1: 'teammate_2'})

    # # # Orange side
    or_teammates_list_v1 = df_reduced.loc[df_reduced.color == 'orange', ['ballchasing_id', 'platform_id']] \
        .merge(or_side.drop('team', axis=1))

    or_teammates_ex = or_teammates_list_v1.explode('id_list').reset_index(drop=True)

    or_teammates_list_v2 = or_teammates_ex[or_teammates_ex.id_list != or_teammates_ex.platform_id] \
        .groupby(['ballchasing_id', 'platform_id'])['id_list'] \
        .apply(list) \
        .reset_index()

    or_teammates = pd.concat([or_teammates_list_v2.loc[:, ['ballchasing_id', 'platform_id']],
                              or_teammates_list_v2.id_list.apply(pd.Series)], axis=1) \
        .rename(columns={0: 'teammate_1', 1: 'teammate_2'})

    # # # Merge both sides
    teammates = pd.concat([or_teammates, bl_teammates])

    # # Opposition Dataframe
    # # # Blue side
    bl_as_opponent_series = bl_side.id_list.apply(pd.Series)

    bl_as_opponent = bl_side \
        .merge(bl_as_opponent_series, left_index=True, right_index=True) \
        .drop('id_list', axis=1) \
        .rename(columns={0: 'opponent_1', 1: 'opponent_2', 2: 'opponent_3', 'team': 'opponent_team'}) \
        .replace({'color': {'blue': 'orange'}})

    # # # Orange side
    or_as_opponent_series = or_side.id_list.apply(pd.Series)

    or_as_opponent = or_side \
        .merge(or_as_opponent_series, left_index=True, right_index=True) \
        .drop('id_list', axis=1) \
        .rename(columns={0: 'opponent_1', 1: 'opponent_2', 2: 'opponent_3', 'team': 'opponent_team'}) \
        .replace({'color': {'orange': 'blue'}})

    # # # Merge both sides
    opps = pd.concat([or_as_opponent, bl_as_opponent])

    # Merge principal dataframe with teammates and opponents ones
    dataframe = dataframe.merge(teammates, how='outer').merge(opps)

    # Remove no longer needed variables
    dataframe = dataframe.drop(['ballchasing_id', 'name', 'color'], axis=1)

    # Change overtime and MVP column to numeric (better format for further exploitation)
    dataframe.overtime = np.where(dataframe.overtime, 1, 0)
    dataframe.core_mvp = np.where(dataframe.core_mvp, 1, 0)

    if export_players_db:
        players_db.to_csv('../../data/retrieved/players_db.csv', encoding='utf8', index=False)

    return dataframe, players_db


if __name__ == '__main__':
    REF_DATE_STR = '2021-10-08 06:00:00+00:00'  # Very first day of RLCS 2021-22
    data, my_players_db = treatment_by_players(REF_DATE_STR)
