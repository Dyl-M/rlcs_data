# ballchasing_ML

Machine Learning around replays uploaded on [ballchasing.com](https://ballchasing.com/).

Introduction
-------------

> Rocket League is a vehicular soccer video game developed and published by Psyonix. The game was first released for Microsoft Windows and PlayStation 4 in July 2015, with ports for Xbox One and Nintendo Switch being released later on. In June 2016, 505 Games began distributing a physical retail version for PlayStation 4 and Xbox One, with Warner Bros. Interactive Entertainment taking over those duties by the end of 2017. Versions for macOS and Linux were also released in 2016, but support for their online services was dropped in 2020. The game went free-to-play in September 2020. Rocket League is a vehicular soccer video game developed and published by Psyonix.

[Wikipedia - Rocket League](https://en.wikipedia.org/wiki/Rocket_League "Wikipedia - Rocket League")

The **`ballchasing_ML`** project aims to pursue the opening of analytical possibilities to the community by using [ballchasing.com](https://ballchasing.com/) database, a website gathering Rocket League games&#39; replays. &quot;ML&quot; stands here for &quot;Machine Learning&quot;, because my first works around this database will be:

* To collect all the data avaible on the website and trying to structure them as well as possible as frist short term goal.
* Then, to create machine learning models able to predict games&#39; outcome as mid term goal.

The long term goal would be to implement machine learning predictions as a game feature (through a mod), but there would be several barriers to overcome before getting there.

* The first is to define the purpose of such a tool, because at the moment of writing this ReadMe, there are two things that I don&#39;t know:
	1. Can I model such a problem effectively?
	2. And in the end, is it really interesting to predict games&#39; outcome?

	Only some exploration of the problem will tell me.

* And the second obstacle is technical, since I have no idea how to make a mod for this game and how to possibly implement this idea.

Current in-game features
-------------

*None so far.*

Tasks done
-------------

*None so far.*

Tasks in progress / planned
-------------

**Work in progress:**

* Data retrieving (`api_call.py`).

**Planned:**

* Data rearrangement (no file associated yet).
* Modeling (no file associated yet).

Repository structure
-------------

Elements followed by `(IGNORED)` are kept ignored / hidden by git for privacy purpose.

```
├── data
│   ├── my_token.txt (IGNORED)
│   └── seasons.json
│
├── src
│   └── api_calls.py
│
├── .deepsource.toml
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

Regarding `my_token.txt`:
* You can generate your personal token on [ballchasing.com](https://ballchasing.com/) by connecting to your Steam 
  account.
* Then write this token into a text file like I did, it will be read to run API calls.

External information
-------------

Codes are reviewed by the [DeepSource](https://deepsource.io/) bot.