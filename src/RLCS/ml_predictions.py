# -*- coding: utf-8 -*-

import ml_formatting
import ml_training

import numpy as np
import pandas as pd
import tensorflow as tf
import warnings

from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=UserWarning)

"""File Information

@file_name: ml_predictions.py
@author: Dylan "dyl-m" Monfret
"""

"GLOBAL"

REF_DATE_STR = '2021-10-08 06:00:00+00:00'  # Very first day of RLCS 2021-22
REF_DATE = datetime.strptime(REF_DATE_STR, '%Y-%m-%d %H:%M:%S%z')

"OPTIONS"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

"FUNCTION"


def generate_match_df(region: str, split: str, event: str, phase: str, stage: str, stage_round: str,
                      match_date_str: str):
    """Create a pandas dataframe with match information suitable for sample generation
    :param region: RLCS region available ('Europe', 'North America', 'Oceania', ..., 'World' for international LAN)
    :param split: part of the season (Fall, Winter, Spring and "Summer" for World Finals)
    :param event: event name (Regional 1-3, Major Tiebreaker, Major, Worlds)
    :param phase: event phase (Main Event, Closed Qualifier, Wildcard stage, etc.)
    :param stage: phase stage (Groups/Swiss, Playoffs)
    :param stage_round: current round (Round 1, Semi Final, Lower Round 1, Grand Final, etc.)
    :param match_date_str: match date as character string
    :return match_df: pandas dataframe with match information
    """
    rlcs_start_date = datetime.strptime('2021-10-08 06:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
    match_date = datetime.strptime(match_date_str, '%Y-%m-%d %H:%M:%S%z')
    time_diff = (match_date - rlcs_start_date) / timedelta(days=1)

    match_df = pd.concat([pd.DataFrame([{'region': region,
                                         'split': split,
                                         'event': event,
                                         'phase': phase,
                                         'stage': stage,
                                         'round': stage_round,
                                         'since_ref_date': time_diff}])] * 6, ignore_index=True)
    return match_df


def generate_virtual_data(missing_ids: list, ref_df: pd.DataFrame, region: str, seed: int = 42069):
    """Generate fake data for missing players in retrieved data (with regional quantile-0.25 by score)
    :param missing_ids: missing players ID
    :param ref_df: reference dataset generated with 'ml_formatting.treatment_by_players' function
    :param region: tournament region to filter ref_df
    :param seed: random seed (and here tournament team seed)
    :return generated_data: generated dataframe.
    """
    n_missing = len(missing_ids)  # Number of line to keep == n missing players

    to_drop = ['region', 'split', 'event', 'phase', 'stage', 'round', 'since_ref_date', 'team', 'platform_id',
               'teammate_1', 'teammate_2', 'opponent_team', 'opponent_1', 'opponent_2', 'opponent_3']

    most_used = ['car_name', 'camera_fov', 'camera_height', 'camera_pitch', 'camera_distance', 'camera_stiffness',
                 'camera_swivel_speed', 'camera_transition_speed', 'steering_sensitivity']

    # Keep regional quantile-0.25 by score
    regional_data = ref_df.loc[(ref_df.region == region) &
                               (ref_df.core_score <= ref_df.core_score.quantile(0.25))].drop(to_drop, axis=1)

    # Retrieve the most used parameters
    most_used_data = pd.concat([regional_data.loc[:, most_used].mode()] * n_missing).reset_index(drop=True)
    most_used_data['platform_id'] = missing_ids

    # Generate fake data based on quantile-0.25 by score
    # # Get mean, standard deviation, median and quantile-0.75 for each statistic
    numeric_data = regional_data.drop(most_used, axis=1)
    numeric_dis = pd.concat([pd.DataFrame(numeric_data.mean()),
                             pd.DataFrame(numeric_data.std())], axis=1).set_axis(['mean_', 'std_'], axis=1)

    # Simulations: Gaussian simulations or mean if simulation end up negative
    np.random.seed(seed)
    numeric_dis['simu_1'] = np.random.normal(numeric_dis.mean_, numeric_dis.std_)
    numeric_dis.simu_1 = np.where(numeric_dis.simu_1 < 0, numeric_dis.mean_, numeric_dis.simu_1)
    numeric_dis.simu_1 = np.where(numeric_dis.simu_1 > numeric_dis.mean_, numeric_dis.mean_, numeric_dis.simu_1)

    np.random.seed(seed * seed)
    numeric_dis['simu_2'] = np.random.normal(numeric_dis.mean_, numeric_dis.std_)
    numeric_dis.simu_2 = np.where(numeric_dis.simu_2 < 0, numeric_dis.mean_, numeric_dis.simu_2)
    numeric_dis.simu_2 = np.where(numeric_dis.simu_2 > numeric_dis.mean_, numeric_dis.mean_, numeric_dis.simu_2)

    np.random.seed(seed * seed * seed)
    numeric_dis['simu_3'] = np.random.normal(numeric_dis.mean_, numeric_dis.std_)
    numeric_dis.simu_3 = np.where(numeric_dis.simu_3 < 0, numeric_dis.mean_, numeric_dis.simu_3)
    numeric_dis.simu_3 = np.where(numeric_dis.simu_3 > numeric_dis.mean_, numeric_dis.mean_, numeric_dis.simu_3)

    # Change simulated stats dataframe format / Transposition and keep n_missing rows
    numeric_dis = numeric_dis.drop(['mean_', 'std_'], axis=1).T.reset_index(drop=True).iloc[0:n_missing, :]

    # Concat everything
    generated_data = pd.concat([most_used_data, numeric_dis], axis=1)

    return generated_data


def generate_data(players_ids: list, ref_df: pd.DataFrame):
    """Create players data based on average statistics with recent and most used settings
    :param players_ids: players ID
    :param ref_df: reference dataset generated with 'ml_formatting.treatment_by_players' function
    :return generated_data:
    """
    match_df_cols = ['region', 'split', 'event', 'phase', 'stage', 'round', 'since_ref_date']
    ordered_cols = ref_df.columns.to_list()
    valid_df = ref_df.loc[ref_df.platform_id.isin(players_ids)]

    # Most frequent car
    most_used = valid_df.loc[:, ['platform_id', 'car_name']] \
        .groupby('platform_id', as_index=False) \
        .agg(lambda x: x.value_counts().index[0])

    # Last used settings
    last_used = valid_df.loc[:, ['platform_id', 'since_ref_date', 'camera_fov', 'camera_height', 'camera_pitch',
                                 'camera_distance', 'camera_stiffness', 'camera_swivel_speed',
                                 'camera_transition_speed', 'steering_sensitivity']] \
        .sort_values('since_ref_date', ascending=False).groupby('platform_id', as_index=False).first() \
        .drop('since_ref_date', axis=1)

    # Numerical columns
    to_throw = match_df_cols + most_used.columns.to_list() + last_used.columns.to_list()
    stats_avg_cols = ['platform_id'] + [col for col in ordered_cols if col not in to_throw][:-6]
    stats_average = valid_df.loc[:, stats_avg_cols].groupby('platform_id', as_index=False).mean()

    # Merge everything
    generated_data = most_used.merge(last_used).merge(stats_average)

    return generated_data


def generate_team_sample(team_dict: dict, ref_df: pd.DataFrame, region: str):
    """Create a team sample
    :param team_dict: team information dictionary
    :param ref_df: reference dataset generated with 'ml_formatting.treatment_by_players' function
    :param region: tournament region (North America, Europe, Oceania, ..., Worlds)
    :return sample: suitable pandas dataframe for prediction with the model implemented in 'ml_training.py' script.
    """
    """
    Suitable team dictionaries format (player_name only serve here as landmark for sample observation):
    {'team': 'TEAM BDS',
     'seed': 1,
     'players': [{'player_name': 'Extra', 'platform_id': 'steam_76561198387068637'},
                 {'player_name': 'M0nkey M00n', 'platform_id': 'steam_76561198823985212'},
                 {'player_name': 'MaRc_By_8', 'platform_id': 'steam_76561198307356688'}]}
    """
    # Init. missing and present players dataframes
    players_present = pd.DataFrame()
    player_missing = pd.DataFrame()

    # Save ordered columns
    to_remove = {'region', 'split', 'event', 'phase', 'stage', 'round', 'since_ref_date', 'win', 'opponent_team',
                 'opponent_1', 'opponent_2', 'opponent_3'}
    ordered_cols = [col_name for col_name in ref_df.columns.to_list() if col_name not in to_remove]

    # Separate known and unknown players
    team_players_ids = [player['platform_id'] for player in team_dict['players']]
    ids_in_dataset = ref_df.loc[ref_df.platform_id.isin(team_players_ids)].platform_id.unique().tolist()
    ids_not_in_dataset = [an_id for an_id in team_players_ids if an_id not in ids_in_dataset]

    if ids_in_dataset:  # For known players
        players_present = generate_data(players_ids=ids_in_dataset,
                                        ref_df=ref_df)

    if ids_not_in_dataset:  # For unknown players
        player_missing = generate_virtual_data(missing_ids=ids_not_in_dataset,
                                               ref_df=ref_df,
                                               region=region,
                                               seed=team_dict['seed'])

    team_df = pd.concat([players_present, player_missing]) \
        .sort_values('core_score', ascending=False) \
        .reset_index(drop=True)

    # Teammates DF
    ordered_players = team_df.platform_id.to_list()

    teammates_dict = {'platform_id': ordered_players, 'id_list': [ordered_players] * len(ordered_players)}
    teammates_df = pd.DataFrame(teammates_dict)
    teammates_df_ex = teammates_df.explode('id_list').reset_index(drop=True)

    teammates_lists = teammates_df_ex[teammates_df_ex.id_list != teammates_df_ex.platform_id] \
        .groupby(['platform_id'])['id_list'] \
        .apply(list) \
        .reset_index()

    teammates = pd.concat([teammates_lists.loc[:, ['platform_id']], teammates_lists.id_list.apply(pd.Series)], axis=1) \
        .rename(columns={0: 'teammate_1', 1: 'teammate_2'})

    # Final sample DF
    team_sample = team_df.merge(teammates)
    team_sample['team'] = team_dict['team']
    team_sample = team_sample[ordered_cols]

    return team_sample


def opposition_sample(team_1_sample: pd.DataFrame, team_2_sample: pd.DataFrame, match_df: pd.DataFrame):
    """Generate a complete sample for prediction with opposition
    :param team_1_sample: first team dataframe
    :param team_2_sample: second team dataframe
    :param match_df: pandas dataframe generated with 'generate_match_df' function
    :return sample: dataframe made for prediction
    """
    ordered_cols = ['platform_id', 'opponent_team', 'opponent_1', 'opponent_2', 'opponent_3']
    team_1_name = team_1_sample.loc[0, 'team']
    team_2_name = team_2_sample.loc[0, 'team']

    opponent_of_team_1 = pd.concat([pd.DataFrame([team_2_sample.platform_id.to_list()],
                                                 columns=['opponent_1', 'opponent_2', 'opponent_3'])] * 3) \
        .reset_index(drop=True)

    opponent_of_team_1['platform_id'] = team_1_sample.platform_id.to_list()
    opponent_of_team_1['opponent_team'] = team_2_name
    opponent_of_team_1 = opponent_of_team_1[ordered_cols]

    opponent_of_team_2 = pd.concat([pd.DataFrame([team_1_sample.platform_id.to_list()],
                                                 columns=['opponent_1', 'opponent_2', 'opponent_3'])] * 3) \
        .reset_index(drop=True)

    opponent_of_team_2['platform_id'] = team_2_sample.platform_id.to_list()
    opponent_of_team_2['opponent_team'] = team_1_name
    opponent_of_team_2 = opponent_of_team_2[ordered_cols]

    opponents = pd.concat([opponent_of_team_1, opponent_of_team_2])
    sample = pd.concat([match_df, pd.concat([team_1_sample, team_2_sample]).merge(opponents)], axis=1)

    return sample


def game_prediction(sample: pd.DataFrame, ref_df: pd.DataFrame, model: tf.keras.Model, n_game_max: int,
                    team_1_dict: dict, team_2_dict: dict):
    """Compute match prediction
    :param sample: a sample created with 'generate_sample' function
    :param ref_df: dataset generated with 'ml_formatting.treatment_by_players' function but wo/ 'win' variable
    :param model: deep learning Keras model suited for the data
    :param n_game_max: maximum game playable (best of 5/7)
    :param team_1_dict: first team information dictionary
    :param team_2_dict: second team information dictionary
    :return: win probabilities and score.
    """
    if n_game_max not in {5, 7}:
        n_game_max = 5

    def bo_score(probability):
        """Compute score teams
        :param probability: win probability
        :return: team score.
        """
        if probability < 0.5:
            return round(n_game_max * probability)
        return n_game_max // 2 + 1

    players = pd.concat([pd.DataFrame(team_1_dict['players']),
                         pd.DataFrame(team_2_dict['players'])])

    new_sample = ml_training.pretreatment(ref_df, sample)
    predictions = model.predict(new_sample)
    sample['probabilities'] = predictions

    player_prob = players \
        .merge(sample.loc[:, ['team', 'platform_id', 'probabilities']]) \
        .sort_values(['team', 'probabilities'], ascending=False) \
        .reset_index(drop=True)

    teams_prob = player_prob.loc[:, ['team', 'probabilities']].groupby('team').sum()
    teams_prob = (teams_prob / teams_prob.sum())
    score = teams_prob.probabilities.apply(bo_score).to_dict()

    return score, teams_prob.to_dict()['probabilities'], player_prob


def game_on(team_1: tuple, team_2: tuple, model: tf.keras.Model, match_df: pd.DataFrame, ref_df: pd.DataFrame,
            n_game_max: int):
    """Generate a match dataframe and compute predictions ('opposition_sample' then 'game_prediction')
    :param team_1: first team tuple (team dict as first item and team dataframe as second)
    :param team_2: second team tuple (team dict as first item and team dataframe as second)
    :param model: deep learning Keras model suited for the data
    :param match_df: pandas dataframe generated with 'generate_match_df' function
    :param ref_df: dataset generated with 'ml_formatting.treatment_by_players' function but wo/ 'win' variable
    :param n_game_max: maximum game playable (best of 5/7)a
    :return team_score, team_prob, player_prob: teams scores predicted, team "win probability", player win probability.
    """
    match = opposition_sample(team_1_sample=team_1[1], team_2_sample=team_2[1], match_df=match_df)

    team_score, team_prob, player_prob = game_prediction(sample=match,
                                                         ref_df=ref_df,
                                                         model=model,
                                                         n_game_max=n_game_max,
                                                         team_1_dict=team_1[0],
                                                         team_2_dict=team_2[0])

    return team_score, team_prob, player_prob


"MAIN"

if __name__ == '__main__':
    DF_GAMES, _ = ml_formatting.treatment_by_players(ref_date_str=REF_DATE_STR)  # Data import and formatting

    # Extract target array
    DATA = DF_GAMES.drop('win', axis=1)
    MY_MODEL = tf.keras.models.load_model('../../models/best_model.h5')

    # Set up the teams
    NRG = {'team': 'THE GENERAL NRG',
           'seed': 1,
           'players': [{'player_name': 'GarrettG', 'platform_id': 'steam_76561198136523266'},
                       {'player_name': 'Squishy', 'platform_id': 'steam_76561198286759507'},
                       {'player_name': 'justin.', 'platform_id': 'steam_76561198299709908'}]}

    RAN = {'team': 'RANDOMS (NORTH AMERICA)',
           'seed': 16,
           'players': [{'player_name': '2Piece', 'platform_id': 'steam_76561198802792967'},
                       {'player_name': 'Chronic', 'platform_id': 'steam_76561198799226283'},
                       {'player_name': 'Night', 'platform_id': 'steam_76561198873647443'}]}

    # Generate match basic information
    REGION = 'North America'
    SPLIT = 'WINTER'
    EVENT = 'Regional 3'
    PHASE = 'Main Event'
    BESTOF5 = 5  # BESTOF7 = 7

    MATCH_DATE = '2022-02-18 18:00:00+00:00'  # Winter Split - North America Regional 3 - First Group match

    MATCH_DF = generate_match_df(region=REGION, split=SPLIT, event=EVENT, phase=PHASE, stage='Groups',
                                 stage_round='Round 3', match_date_str=MATCH_DATE)

    # Generate match sample and then do prediction
    NRG_SAMPLE = generate_team_sample(team_dict=NRG, ref_df=DATA, region=REGION)
    RAN_SAMPLE = generate_team_sample(team_dict=RAN, ref_df=DATA, region=REGION)

    SCORE, _, _ = game_on(team_1=(NRG, NRG_SAMPLE), team_2=(RAN, RAN_SAMPLE), model=MY_MODEL, match_df=MATCH_DF,
                          ref_df=DATA, n_game_max=BESTOF5)

    print(f'SCORE: {SCORE}')
