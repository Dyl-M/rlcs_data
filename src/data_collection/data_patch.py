import pandas as pd

pd.set_option('display.max_columns', None)  # pd.set_option('display.max_rows', None)
pd.set_option('display.width', 225)

"""File Information
@file_name: data_patch.py
@author: Dylan "dyl-m" Monfret
To help me patch RLCS datasets.
"""


def missing_info():
    """A function to check missing value in datasets
    :return: missing cars, missing maps, missing countries.
    """
    missing_car = pd.read_csv('../../data/retrieved/games_by_players.csv', encoding='utf8', low_memory=False)
    missing_car = missing_car.loc[(missing_car.car_name.isna()) & (missing_car.car_id.notna()),
                                  ['game_id', 'player_tag', 'car_name', 'car_id']]

    missing_map = pd.read_csv('../../data/retrieved/main.csv', encoding='utf8', low_memory=False)
    missing_map_id = missing_map.loc[(missing_map.map_id.isna()) & (missing_map.map_name.notna()),
                                     ['map_id', 'map_name']].drop_duplicates()

    missing_map_name = missing_map.loc[(missing_map.map_id.notna()) & (missing_map.map_name.isna()),
                                       ['map_id', 'map_name']].drop_duplicates()

    missing_country = pd.read_csv('../../data/retrieved/players_db.csv', encoding='utf8', low_memory=False)
    missing_country = missing_country.loc[missing_country.player_country.isna()]

    return missing_car, missing_map_id, missing_map_name, missing_country


if __name__ == '__main__':
    CAR, MAP_ID, MAP_NAME, COUNTRY = missing_info()
    print(CAR)
    print(MAP_ID)
    print(MAP_NAME)
    print(COUNTRY)
