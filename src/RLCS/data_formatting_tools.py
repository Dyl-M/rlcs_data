# -*- coding: utf-8 -*-

import pandas as pd

"""File Information

@file_name: data_formatting_tools.py
@author: Dylan "dyl-m" Monfret
"""

"FUNCTIONS"


def display_rosters(pd_dataframe_players: pd.DataFrame, pd_dataframe_general: pd.DataFrame):
    """Display team roster iteratively
    :param pd_dataframe_players: dataframe by players
    :param pd_dataframe_general: dataframe with general information.
    """
    date_df = pd_dataframe_general.loc[:, ['ballchasing_id', 'date']]

    pd_dataframe_players = date_df.merge(pd_dataframe_players).sort_values('date')
    teams = sorted(pd_dataframe_players.name.unique())

    for team in teams:
        df_values = pd_dataframe_players \
            .loc[pd_dataframe_players.name == team, ['p_name', 'p_platform', 'p_platform_id']] \
            .drop_duplicates('p_platform_id', keep='last') \
            .sort_values('p_name', key=lambda col: col.str.lower())

        if df_values.shape[0] != 3:
            print(team)
            print(df_values)
            print('\n')


def sort_alias():
    """Sort alias JSON file by value then by key."""
    import json

    with open('../../data/public/alias.json', 'r', encoding='utf8') as alias_file:
        alias_dict = json.load(alias_file)

    sorted_alias_dict = dict(sorted(alias_dict.items(), key=lambda item: (item[1], item[0])))

    with open('../../data/public/alias.json', 'w', encoding='utf-8') as sorted_alias_file:
        json.dump(sorted_alias_dict, sorted_alias_file)


def do_nothing():
    return None


"MAIN"

if __name__ == "__main__":
    do_nothing()
