# -*- coding: utf-8 -*-

import pandas as pd

"""File Information

@file_name: data_formatting_tools.py
@author: Dylan "dyl-m" Monfret
"""

"FUNCTIONS"


def display_team_name_missing(pd_dataframe_players: pd.DataFrame):
    """Display ballchasing_id where teams names are missing
    :param pd_dataframe_players: dataframe by players.
    """
    import webbrowser
    team_missing = pd_dataframe_players[pd_dataframe_players.name.isna()]['ballchasing_id'].unique().tolist()

    try:
        webbrowser.open(f'https://ballchasing.com/replay/{team_missing[0]}')
        print(team_missing)

    except IndexError:
        print("No missing values.")


def display_team_blue_orange(pd_dataframe_players: pd.DataFrame):
    """Display ballchasing_id where teams names are missing
    :param pd_dataframe_players: dataframe by players.
    """
    team_is_blue = pd_dataframe_players[pd_dataframe_players.name == 'BLUE']['ballchasing_id'].unique().tolist()
    team_is_orange = pd_dataframe_players[pd_dataframe_players.name == 'ORANGE']['ballchasing_id'].unique().tolist()
    team_is_orange_blue = set(team_is_blue + team_is_orange)

    if team_is_orange_blue:
        print(team_is_orange_blue)

    else:
        print("Everything seems good!")


def display_teams_names(pd_dataframe_players: pd.DataFrame):
    """Display teams
    :param pd_dataframe_players: dataframe by players.
    """
    import pprint
    teams = sorted(pd_dataframe_players.name.unique())
    pprint.pprint(teams)


def display_rosters(pd_dataframe_players: pd.DataFrame, pd_dataframe_general: pd.DataFrame, not_three: bool = True):
    """Display team roster iteratively
    :param pd_dataframe_players: dataframe by players
    :param pd_dataframe_general: dataframe with general information
    :param not_three: display rosters if more or less than three players.
    """
    date_df = pd_dataframe_general.loc[:, ['ballchasing_id', 'date', 'split']]
    splits = ['Fall', 'Winter']
    pd_dataframe_players = date_df.merge(pd_dataframe_players).sort_values('date')
    teams = sorted(pd_dataframe_players.name.unique())

    for team in teams:
        print(team)
        print()

        for split in splits:
            df_values = pd_dataframe_players \
                .loc[pd_dataframe_players.split == split] \
                .loc[pd_dataframe_players.name == team, ['p_name', 'p_platform', 'p_platform_id']] \
                .drop_duplicates('p_platform_id', keep='last') \
                .sort_values('p_name', key=lambda col: col.str.lower())

            if not_three:
                if not df_values.empty and df_values.shape[0] != 3:
                    print(split)
                    print(df_values)
                    print()

            else:
                if not df_values.empty:
                    print(split)
                    print(df_values)
                    print()

        print('\n')


def sort_alias():
    """Sort alias JSON file by value then by key."""
    import json

    with open('../../data/public/alias.json', 'r', encoding='utf8') as alias_file:
        alias_dict = json.load(alias_file)

    sorted_alias_dict = dict(sorted(alias_dict.items(), key=lambda item: (item[1], item[0])))

    with open('../../data/public/alias.json', 'w', encoding='utf-8') as sorted_alias_file:
        json.dump(sorted_alias_dict, sorted_alias_file, indent=2)
