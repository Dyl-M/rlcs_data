# RLCS DATA | `rlcs_data`

[![GitHub last commit](https://img.shields.io/github/last-commit/Dyl-M/ballchasing_ML?label=Last%20commit&style=flat-square)](https://github.com/Dyl-M/ballchasing_ML/commits/main)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/w/Dyl-M/ballchasing_ML?label=Commit%20activity&style=flat-square)](https://github.com/Dyl-M/ballchasing_ML/commits/main)
[![DeepSource](https://deepsource.io/gh/Dyl-M/rlcs_data.svg/?label=active+issues&token=w_aZJJfhd5HPPLyXnDJkstmn)](https://deepsource.io/gh/Dyl-M/ballchasing_ML/?ref=repository-badge)
[![DeepSource](https://deepsource.io/gh/Dyl-M/rlcs_data.svg/?label=resolved+issues&token=w_aZJJfhd5HPPLyXnDJkstmn)](https://deepsource.io/gh/Dyl-M/ballchasing_ML/?ref=repository-badge)

[![Twitter Follow](https://img.shields.io/twitter/follow/dyl_m_tweets?label=%40dyl_m_tweets&style=social)](https://twitter.com/dyl_m_tweets)
[![Reddit User Karma](https://img.shields.io/reddit/user-karma/link/dyl_m?label=u%2Fdyl_m&style=social)](https://www.reddit.com/user/Dyl_M)

Data mining and Machine Learning around the Rocket League Championship Series.

Introduction
-------------

> Rocket League is a vehicular soccer video game developed and published by Psyonix. The game was first released for Microsoft Windows and PlayStation 4 in July 2015, with ports for Xbox One and Nintendo Switch being released later on. In June 2016, 505 Games began distributing a physical retail version for PlayStation 4 and Xbox One, with Warner Bros. Interactive Entertainment taking over those duties by the end of 2017. Versions for macOS and Linux were also released in 2016, but support for their online services was dropped in 2020. The game went free-to-play in September 2020.
>
>*[Wikipedia - Rocket League](https://en.wikipedia.org/wiki/Rocket_League "Wikipedia - Rocket League")*

> The **Rocket League Championship Series (RLCS)** is an annual (previously semiannual) Rocket League Esports tournament series produced by Psyonix, the game's developer. It consists of qualification splits in North America, South America, Europe, Oceania, Middle East/North Africa, Asia, and Sub-Saharan Africa, and culminates in a playoff bracket with teams from those regions. The qualification rounds are played as an online round-robin tournament and the finals are played live in different cities. [...]
>
> *[Wikipédia - Rocket League Championship Series](https://en.wikipedia.org/wiki/Rocket_League_Championship_Series)*

The **`rlcs_data`** project aims to pursue the opening of analytical possibilities to the community by using Rocket League replays (`.replay` files) and statistics databases. My first works around those database will be:

* The retrieving and structuring data around and with sites and APIs established by the community.
* The conception of machine learning methodologies and models able to predict RLCS games' outcomes.

To perform these tasks, the APIs of [octane.gg](https://octane.gg/) and [ballchasing.com](https://ballchasing.com/) will be used to retrieve data, as complete as possible. All data will be available on the Kaggle platform through the links below :

* [Rocket League Championship Series 2021-2022](https://www.kaggle.com/dylanmonfret/rlcs-202122).
* Rocket League Championship Series 2022-2023 (SOON).

Repository structure
-------------

Elements followed by `(IGNORED)` are kept ignored / hidden by git for privacy purpose or due to their size.

```
├── .github
│   └── ISSUE_TEMPLATE
│       └── feature_request.md
│
├── cmd (IGNORED)
│
│
├── data
│   ├── archive (IGNORED)
│   │   └── rlcs_2021-22
│   │
│   ├── private (IGNORED)
│   │   ├── my_token.txt
│   │   ├── random_seeds.json
│   │   └── tokens.json
│   │
│   ├── public
│   │   ├── data_coverage.csv
│   │   ├── missing_values.json
│   │   ├── note.txt
│   │   ├── seasons.json
│   │   ├── true_cols.json
│   │   └── winter_major_patch.json
│   │
│   └── retrieved (IGNORED)
│
├── models (IGNORED)
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
│   ├── data_collection
│   │   ├── collection.py
│   │   └── patch.py
│   │
│   ├── machine_learning
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
* No token is required to use the octane.gg API.

External information
-------------

Codes are reviewed by the [DeepSource](https://deepsource.io/) bot.