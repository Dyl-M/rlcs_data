# ballchasing_ML

[![DeepSource](https://deepsource.io/gh/Dyl-M/ballchasing_ML.svg/?label=active+issues&token=w_aZJJfhd5HPPLyXnDJkstmn)
](https://deepsource.io/gh/Dyl-M/ballchasing_ML/?ref=repository-badge) [![DeepSource](https://deepsource.io/gh/Dyl-M/ballchasing_ML.svg/?label=resolved+issues&token=w_aZJJfhd5HPPLyXnDJkstmn)](https://deepsource.io/gh/Dyl-M/ballchasing_ML/?ref=repository-badge)

Machine Learning around replays uploaded on [ballchasing.com](https://ballchasing.com/) and Rocket League Championship Series.

Introduction
-------------

> Rocket League is a vehicular soccer video game developed and published by Psyonix. The game was first released for Microsoft Windows and PlayStation 4 in July 2015, with ports for Xbox One and Nintendo Switch being released later on. In June 2016, 505 Games began distributing a physical retail version for PlayStation 4 and Xbox One, with Warner Bros. Interactive Entertainment taking over those duties by the end of 2017. Versions for macOS and Linux were also released in 2016, but support for their online services was dropped in 2020. The game went free-to-play in September 2020.
> 
>[Wikipedia - Rocket League](https://en.wikipedia.org/wiki/Rocket_League "Wikipedia - Rocket League")

The **`ballchasing_ML`** project aims to pursue the opening of analytical possibilities to the community by using [ballchasing.com](https://ballchasing.com/) database, a website gathering Rocket League games&#39; replays. &quot;ML&quot; stands here for &quot;Machine Learning&quot;, because my first works around this database will be:

* To collect all the data available on the website and trying to structure them as well as possible as first short term goal.
* Then, to create machine learning models able to predict games' outcomes.

First tries will be processed on RLCS 2021 - 2022 games, subject to games' availability. Entire datasets are available on Kaggle to the following link: [Rocket League Championship Series 2021-2022 on Kaggle](https://www.kaggle.com/dylanmonfret/rlcs-202122).

Repository structure
-------------

Elements followed by `(IGNORED)` are kept ignored / hidden by git for privacy purpose or due to their size.

```
├── data
│   ├── private (IGNORED)
│   │   └── my_token.txt
│   │
│   ├── public
│   │   ├── alias.json
│   │   ├── missing_values.json
│   │   ├── note.txt
│   │   ├── patch.json
│   │   ├── pre_patch.json
│   │   └── seasons.json
│   │
│   └── retrieved (IGNORED)
│       ├── by_players.csv 
│       ├── by_teams.csv
│       ├── general.csv
│       ├── groups.csv
│       ├── players_db.csv
│       ├── pre_dataset.json
│       └── raw.json
│
├── models (IGNORED, temporarily?)
│   ├── tuning_results
│   │   └── evaluation.csv
│   │
│   ├── best_model.h5
│   └── tmp_mdl.h5
│
├── notebooks
│   └── rlcs-2021-22-demo.ipynb
│
├── reports
│   └── figures (IGNORED)
│
├── src
│   ├── RLCS
│   │   ├── data_collection.py
│   │   ├── data_collection_tools.py
│   │   ├── data_formatting.py
│   │   ├── data_formatting_tools.py
│   │   ├── ml_formatting.py
│   │   ├── ml_predictions.py
│   │   └── ml_training.py
│   │
│   └── test.py
│
├── .deepsource.toml
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

Regarding `my_token.txt`:

* You can generate your personal token on [ballchasing.com](https://ballchasing.com/) by connecting to your Steam account.
* Then write this token into a text file like I did, it will be read to run API calls.

External information
-------------

Codes are reviewed by the [DeepSource](https://deepsource.io/) bot.