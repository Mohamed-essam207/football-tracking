<img width="1917" height="1078" alt="Screenshot 2026-09-26 212620" src="https://github.com/user-attachments/assets/233af254-2f9e-44e1-a219-942b8046d344" />
# Football tracking + possession + xG

Detects and tracks players and the ball in a football video, splits the players into two teams
automatically from their jersey colors, tracks ball possession, counts passes, and (optionally)
detects shots and estimates a heuristic xG. Built on top of
[tryolabs/soccer-video-analytics](https://github.com/tryolabs/soccer-video-analytics) (MIT license).

## Quick start

Everything runs in **`football_possession_tracking.ipynb`** on Google Colab — no local setup
needed. Open it in Colab (button below), run the cells in order, and pick your video when
prompted.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mohamed-essam207/football-tracking/blob/main/football_possession_tracking.ipynb)


## What's in here

| file | what it does |
|---|---|
| `football_possession_tracking.ipynb` | the notebook — run this |
| `run.py` | the tracking + possession + passes pipeline |
| `auto_team_classifier.py` | splits players into two teams from jersey colors (k-means); referee/goalkeepers become "Other" |
| `ultralytics_detector.py` | lets the pipeline use models trained with `ultralytics` (YOLOv8/11) |
| `custom_board.py` | the possession and passes boards drawn on the video |
| `viz_tweaks.py` | high quality H.264 output, upscaling of small videos, thinner player boxes |
| `compute_xg.py` | detects shots from the ball's speed/direction and estimates a heuristic xG (optional) |
| `xg_overlay.py` | draws the shots + a running xG total onto the video (optional) |

## Models

You need your own trained weights: a ball detector, and optionally a goal detector and/or a
player detector (all `ultralytics`-format `.pt` files). They are **not** included in this repo
(see `.gitignore`) — keep them on Google Drive and point the notebook's settings cell at them, or
attach them to a [GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
if you want them versioned alongside the code.

## xG — important caveat

The xG in this project is a **hand-set geometric heuristic** (distance + angle to goal through a
logistic curve), not a model fitted to real shot outcomes — there's no labeled dataset of shots
behind it. Treat the numbers as a relative sense of shot quality within a video, not literal goal
probabilities. See the docstring in `compute_xg.py` for the full explanation and the tunable
parameters.

## License

The base pipeline is by [Tryolabs](https://github.com/tryolabs/soccer-video-analytics), MIT
licensed. This project also uses [`ultralytics`](https://github.com/ultralytics/ultralytics),
which is **AGPL-3.0** — check its terms before using this commercially or distributing it as a
closed-source service.
