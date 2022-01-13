# -*- coding: utf-8 -*-

import requests

"""File Informations

@file_name: test.py
@author: Dylan "dyl-m" Monfret
To test things / backup functions
"""

with open('../data/private/my_token.txt', 'r', encoding='utf8') as token_file:
    my_token = token_file.read()

"FUNCTIONS"


def get_my_uploads(token: str):
    """Get my replays
    :param token: API token
    :return replays: list of replays
    """
    replays = []
    uri = f'https://ballchasing.com/api/replays/?uploader=me&count=200'
    headers = {'Authorization': token}
    request = requests.get(uri, headers=headers)
    page = request.json()
    replays += [f'https://ballchasing.com/replay/{replay["id"]}' for replay in page["list"]]

    try:
        while page['next']:
            uri = page['next']
            headers = {'Authorization': token}
            request = requests.get(uri, headers=headers)
            page = request.json()
            replays += [f'https://ballchasing.com/replay/{replay["id"]}' for replay in page["list"]]

    except KeyError:
        pass

    return replays


"MAIN"

if __name__ == '__main__':
    pass
