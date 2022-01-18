# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from datetime import datetime

"""File Information

@file_name: ml_formatting.py
@author: Dylan "dyl-m" Monfret

Treatment to apply on .csv files for ML model conception.
"""

"FUNCTIONS"


def treatment_by_teams(ref_date_str):
    """Pretreatment pipeline to build a dataframe set for models by teams.

    :param ref_date_str: A reference date as string, to put a weight on matches based to how old those games are.
    :return final_dataset: Dataset formatted for modeling.
    """
    # Reference date to game datetime
    ref_date = datetime.strptime(ref_date_str, '%Y-%m-%d %H:%M:%S%z')

    # Dataframe imports
    teams_df = pd.read_csv('../../data/retrieved/by_teams.csv', encoding='utf8')
    general_df = pd.read_csv('../../data/retrieved/general.csv', encoding='utf8')

    # Filter sides (orange / blue)
    blue_side = teams_df[teams_df['color'] == 'blue'] \
        .drop(['color'], axis=1) \
        .add_prefix('blue_') \
        .rename(columns={'blue_ballchasing_id': 'ballchasing_id'})

    orange_side = teams_df[teams_df['color'] == 'orange'] \
        .drop(['color'], axis=1) \
        .add_prefix('orange_') \
        .rename(columns={'orange_ballchasing_id': 'ballchasing_id'})

    # Join both sides
    match_results = blue_side.merge(orange_side)

    # Treatment for individual match results
    bo_df = general_df.loc[:, ['ballchasing_id', 'bo_id', 'region', 'split', 'event', 'phase', 'stage', 'round',
                               'map_name', 'duration', 'overtime', 'overtime_seconds', 'date']]

    match_results = bo_df.merge(match_results)

    match_results.loc[match_results.blue_core_goals < match_results.orange_core_goals, 'win'] = 'orange'
    match_results.loc[match_results.blue_core_goals > match_results.orange_core_goals, 'win'] = 'blue'

    # Changing 'date' to a time delta with the reference date
    match_results['date'] = pd.to_datetime(match_results['date'], utc=True)
    match_results.date = (ref_date - match_results.date) / np.timedelta64(1, 'D')
    match_results = match_results.rename(columns={'date': 'since_ref_date'})

    # Group-by for stats by match up
    match_results_mean = match_results.groupby('bo_id', as_index=False).mean().drop(['duration', 'overtime'], axis=1)
    match_results_sum = match_results.groupby('bo_id', as_index=False).sum()[['bo_id', 'duration', 'overtime']]

    bo_results = match_results \
        .groupby('bo_id', as_index=False)[['region', 'split', 'event', 'phase', 'stage', 'round', 'blue_name',
                                           'orange_name', 'win']] \
        .agg(pd.Series.mode)

    # Set the maximum BO matches
    bo_results.loc[bo_results.stage == 'Swiss', 'bo_type'] = 'best_of_5'
    bo_results.loc[bo_results.stage != 'Swiss', 'bo_type'] = 'best_of_7'

    # Count games played by BO
    game_count = match_results.groupby('bo_id', as_index=False) \
        .count()[['bo_id', 'ballchasing_id']] \
        .rename(columns={'ballchasing_id': 'n_game'})

    # Global merge / 'win' column put at the end
    final_dataset = bo_results.merge(game_count).merge(match_results_sum).merge(match_results_mean)
    wins_df = final_dataset[['bo_id', 'win']]
    final_dataset = final_dataset.drop('win', axis=1).merge(wins_df)

    return final_dataset
