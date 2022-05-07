# -*- coding: utf-8 -*-

import datetime
import json
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

"""File Information
@file_name: ml_formatting.py
@author: Dylan "dyl-m" Monfret
Treatment to apply on .csv files for ML model conception.
"""

"FUNCTIONS"


def treatment_by_players(ref_date_str: str = '2021-10-08 06:00:00+00:00'):
    """Pretreatment pipeline to build a dataframe set for models by players
    :param ref_date_str: A reference date as string, to put a weight on matches based to how old those games are,
    RLCS 21-22 starting date by default (OCE Fall Regional 1 Inv. Qualifier)
    :return: Dataset formatted for modeling and players name and ID database.
    """

    def teammates_opponents(input_df: pd.DataFrame, team_color: str):
        """Format input data to extract teammates and opposition
        :param input_df: input dataframe
        :param team_color: team color
        :return: teammates and opposition.
        """
        df_reduced = input_df.loc[:, ['game_id', 'color', 'team_id', 'player_id', 'core_score']]  # Create reduced DF

        # Split blue / orange side and group players ID into list
        df_side = df_reduced.loc[df_reduced.color == team_color] \
            .rename(columns={'player_id': 'id_list'}) \
            .sort_values(['game_id', 'core_score'], ascending=False) \
            .groupby(['game_id', 'color', 'team_id'])['id_list'] \
            .apply(list) \
            .reset_index()

        # Teammates Dataframe
        teammates_list_v1 = df_reduced.loc[df_reduced.color == team_color, ['game_id', 'player_id']] \
            .merge(df_side.drop('team_id', axis=1))

        teammates_ex = teammates_list_v1.explode('id_list').reset_index(drop=True)

        teammates_list_v2 = teammates_ex[teammates_ex.id_list != teammates_ex.player_id] \
            .groupby(['game_id', 'player_id'])['id_list'].apply(list).reset_index()

        as_teammates = pd.concat([teammates_list_v2.loc[:, ['game_id', 'player_id']],
                                  teammates_list_v2.id_list.apply(pd.Series)], axis=1) \
            .rename(columns={0: 'teammate_1', 1: 'teammate_2'})

        # Opposition Dataframe

        if team_color == 'blue':
            opp_color = 'orange'
        else:
            opp_color = 'blue'

        as_opponent_series = df_side.id_list.apply(pd.Series)

        as_opponent = df_side \
            .merge(as_opponent_series, left_index=True, right_index=True) \
            .drop('id_list', axis=1) \
            .rename(columns={0: 'opponent_1', 1: 'opponent_2', 2: 'opponent_3', 'team_id': 'opponent_team'}) \
            .replace({'color': {team_color: opp_color}})

        return as_teammates, as_opponent

    def most_used_settings(input_df: pd.DataFrame):
        """
        :param input_df: input dataframe
        :return:
        """
        settings = input_df.loc[:, ['steering_sensitivity', 'camera_fov', 'camera_height', 'camera_pitch',
                                    'camera_distance', 'camera_stiffness', 'camera_swivel_speed',
                                    'camera_transition_speed']].mean().round(1).to_dict()

        settings_cat = input_df.loc[:, ['platform', 'car_id']].mode().to_dict('records')[0]
        settings.update(settings_cat)

        return settings

    # Reference date to game datetime
    ref_date = datetime.datetime.strptime(ref_date_str, '%Y-%m-%d %H:%M:%S%z')

    # Dataframe imports
    game_df = pd.read_csv('../../data/retrieved/games_by_players.csv', encoding='utf8', low_memory=False)
    main_df = pd.read_csv('../../data/retrieved/main.csv', encoding='utf8', low_memory=False)
    player_df = pd.read_csv('../../data/retrieved/players_db.csv', encoding='utf8', low_memory=False)

    with open('../../data/public/true_cols.json', 'r', encoding='utf8') as tru_cols:
        col_order = json.load(tru_cols)['ordering']['ml_formatting']  # Import columns order

    main_df = main_df[main_df.game_id.notna()].reset_index(drop=True)  # Drop in main_df where 'game_id' is NaN

    # Keep relevant features from each dataset
    main_df = main_df.loc[:, ['game_id', 'game_date', 'event', 'event_split', 'event_region', 'event_phase', 'stage',
                              'stage_is_lan', 'stage_is_qualifier', 'location_country', 'match_round',
                              'match_format', 'overtime']]

    game_df = game_df.drop(['player_tag', 'platform_id', 'car_name'], axis=1)
    player_df = player_df.loc[:, ['player_id', 'player_country']]

    # Merge datasets
    dataframe = main_df.merge(game_df).merge(player_df)

    # Changing 'game_date' to a time delta with the reference date and reorder columns
    dataframe.game_date = pd.to_datetime(dataframe.game_date, utc=True)
    dataframe.game_date = (dataframe.game_date - ref_date) / np.timedelta64(1, 'D')
    dataframe = dataframe.rename(columns={'game_date': 'since_ref_date'})
    dataframe = dataframe[col_order]

    # Filter where this new field is empty
    dataframe = dataframe.loc[dataframe.since_ref_date.notna()]

    # Add opponents and teammates as features
    bl_team, bl_oppo = teammates_opponents(input_df=dataframe, team_color='blue')  # Blue side
    or_team, or_oppo = teammates_opponents(input_df=dataframe, team_color='orange')  # Orange side
    teammates = pd.concat([bl_team, or_team]).reset_index(drop=True)  # Merge both sides in team POV
    opposition = pd.concat([bl_oppo, or_oppo]).reset_index(drop=True)  # Merge both sides in opposition POV
    dataframe = dataframe.merge(teammates, how='outer').merge(opposition)  # Merge with principal dataframe
    dataframe = dataframe.drop(['game_id', 'color'], axis=1)  # Remove no longer needed variables

    # Change winner, overtime and MVP columns to numeric (better format for further exploitation)
    dataframe.advanced_mvp = np.where(dataframe.advanced_mvp, 1, 0)
    dataframe.winner = np.where(dataframe.winner, 1, 0)
    dataframe.overtime = np.where(dataframe.overtime, 1, 0)

    # Change some column type and fill NaN
    dataframe.car_id = dataframe.car_id.fillna(-1)
    dataframe.car_id = dataframe.car_id.astype(int)
    dataframe.car_id = dataframe.car_id.astype(str)

    dataframe.location_country = dataframe.location_country.fillna('no_country')
    dataframe.advanced_rating = dataframe.advanced_rating.fillna(0)
    dataframe.car_id = dataframe.car_id.replace('-1', np.nan)

    common_settings = most_used_settings(input_df=dataframe)

    for k, v in common_settings.items():
        dataframe[k] = dataframe[k].fillna(v)

    return dataframe


if __name__ == '__main__':
    DF = treatment_by_players()
    print(DF)
