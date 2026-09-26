"""
Football possession + tracking on top of tryolabs/soccer-video-analytics.

  python run.py --video videos/match.mp4 --model models/ball.pt
  python run.py --video videos/match.mp4 --model models/ball.pt --player-model models/player.pt

What it does per frame: detect players + ball, track them (norfair), split the players into
two teams automatically (jersey colors), decide who has the ball, and write a video with the
tracking boxes and a possession board. The final possession percentages are printed at the end.
"""
import argparse
import csv
import json
import os

import cv2

# Colab / servers have no GUI: norfair calls these after writing frames
cv2.waitKey = lambda *a, **k: -1
cv2.destroyAllWindows = lambda *a, **k: None

import numpy as np
import PIL
import PIL.Image
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean

from auto_team_classifier import AutoTeamClassifier
from custom_board import draw_passes_board, draw_possession_board
from inference import Converter, InertiaClassifier, YoloV5
from run_utils import (
    get_ball_detections,
    get_main_ball,
    get_player_detections,
    update_motion_estimator,
)
from soccer import Match, Player, Team
from soccer.draw import AbsolutePath
from ultralytics_detector import UltralyticsDetector
from viz_tweaks import HQWriter, draw_goal_boxes, prepare_frame, thin_boxes

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="videos/soccer_possession.mp4", type=str, help="Input video")
parser.add_argument("--model", default="models/ball.pt", type=str, help="Ball detector (ultralytics .pt)")
parser.add_argument("--out-dir", default="outputs", type=str, help="Folder for the output video")
parser.add_argument(
    "--possession",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Draw tracking + possession board (on by default)",
)
parser.add_argument(
    "--passes",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Draw the passes count board under the possession board (on by default)",
)
parser.add_argument("--warmup-frames", default=40, type=int, help="Frames sampled to learn the team colors")
parser.add_argument("--ball-imgsz", default=1280, type=int, help="Inference size of the ball model")
parser.add_argument("--ball-conf", default=0.10, type=float, help="Minimum confidence of the ball model")
parser.add_argument(
    "--player-model",
    default=None,
    type=str,
    help="Optional: your own player detector (ultralytics .pt). Default: pretrained YOLOv5x (COCO 'person')",
)
parser.add_argument(
    "--player-classes",
    default=None,
    type=str,
    help="Comma-separated class names to keep from --player-model (default: every class except the ball)",
)
parser.add_argument("--player-conf", default=0.35, type=float, help="Minimum confidence for --player-model")
parser.add_argument("--player-imgsz", default=1280, type=int, help="Inference size for --player-model")
parser.add_argument(
    "--goal-model",
    default=None,
    type=str,
    help="Optional: goal detector (ultralytics .pt). Draws the goal boxes and saves them to a CSV",
)
parser.add_argument("--goal-conf", default=0.30, type=float, help="Minimum confidence for --goal-model")
parser.add_argument("--goal-imgsz", default=1280, type=int, help="Inference size for --goal-model")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
video = Video(input_path=args.video, output_path=args.out_dir)
fps = video.video_capture.get(cv2.CAP_PROP_FPS)
thin_boxes()
writer = HQWriter(video, fps)

# Object detectors
if args.player_model:
    player_detector = UltralyticsDetector(
        model_path=args.player_model, conf=args.player_conf, imgsz=args.player_imgsz
    )
    keep_classes = (
        {c.strip().lower() for c in args.player_classes.split(",") if c.strip()}
        if args.player_classes
        else None
    )

    def detect_players(frame):
        df = player_detector.predict(frame)
        names = df["name"].astype(str).str.lower()
        df = df[names.isin(keep_classes)] if keep_classes else df[~names.str.contains("ball")]
        return Converter.DataFrame_to_Detections(df)

else:
    player_detector = YoloV5()

    def detect_players(frame):
        return get_player_detections(player_detector, frame)


ball_detector = UltralyticsDetector(
    model_path=args.model, imgsz=args.ball_imgsz, conf=args.ball_conf
)

# Optional goal detector (for the goal boxes on the video and the CSV used later for xG work)
goal_detector = None
goal_rows = []
ball_rows = []  # per-frame ball position + team in possession, used later for shot/xG detection
if args.goal_model:
    goal_detector = UltralyticsDetector(
        model_path=args.goal_model, conf=args.goal_conf, imgsz=args.goal_imgsz
    )

# Automatic team classifier: learns the 2 jersey colors from the video itself
auto_classifier = AutoTeamClassifier(team_names=["Team A", "Team B"])
auto_classifier.fit_from_video(args.video, detect_players, n_frames=args.warmup_frames)

# Add inertia to classifier
classifier = InertiaClassifier(classifier=auto_classifier, inertia=20)


def text_color_for(bgr):
    luminance = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


# Teams and Match (colors = the real jersey colors found in the video)
color_a, color_b = auto_classifier.team_colors()
team_a = Team(
    name="Team A",
    abbreviation="TMA",
    color=color_a,
    board_color=color_a,
    text_color=text_color_for(color_a),
)
team_b = Team(
    name="Team B",
    abbreviation="TMB",
    color=color_b,
    board_color=color_b,
    text_color=text_color_for(color_b),
)
teams = [team_a, team_b]
match = Match(home=team_a, away=team_b, fps=fps)

# Tracking
player_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=250,
    initialization_delay=3,
    hit_counter_max=90,
)

ball_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=150,
    initialization_delay=5,
    hit_counter_max=2000,
)
motion_estimator = MotionEstimator()
coord_transformations = None

# Paths
path = AbsolutePath()

# Get Counter img
for i, frame in enumerate(video):
    frame = prepare_frame(frame)

    # Goal detections (optional)
    goal_df = None
    if goal_detector is not None:
        goal_df = goal_detector.predict(frame)
        for r in goal_df.to_dict("records"):
            goal_rows.append(
                [i, r["name"], round(float(r["confidence"]), 4)]
                + [round(float(r[k]), 1) for k in ("xmin", "ymin", "xmax", "ymax")]
                + [frame.shape[1], frame.shape[0]]
            )

    # Get Detections
    players_detections = detect_players(frame)
    ball_detections = get_ball_detections(ball_detector, frame)
    detections = ball_detections + players_detections

    # Update trackers
    coord_transformations = update_motion_estimator(
        motion_estimator=motion_estimator,
        detections=detections,
        frame=frame,
    )

    player_track_objects = player_tracker.update(
        detections=players_detections, coord_transformations=coord_transformations
    )

    ball_track_objects = ball_tracker.update(
        detections=ball_detections, coord_transformations=coord_transformations
    )

    player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
    ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)

    player_detections = classifier.predict_from_detections(
        detections=player_detections,
        img=frame,
    )

    # Match update
    ball = get_main_ball(ball_detections)
    players = Player.from_detections(detections=players_detections, teams=teams)
    match.update(players, ball)

    center = ball.center if ball else None  # (x, y) in the (possibly resized) frame, or None
    bx, by = (float(center[0]), float(center[1])) if center is not None else ("", "")
    ball_rows.append([i, bx, by, match.team_possession.name if match.team_possession else ""])

    # Draw
    frame = PIL.Image.fromarray(frame)
    frame = draw_goal_boxes(frame, goal_df)

    if args.possession:
        frame = Player.draw_players(
            players=players, frame=frame, confidence=False, id=True
        )

        frame = path.draw(
            img=frame,
            detection=ball.detection,
            coord_transformations=coord_transformations,
            color=match.team_possession.color,
        )

        frame = draw_possession_board(frame, match)
        if args.passes:
            frame = draw_passes_board(frame, match)
        if match.closest_player:
            frame = match.closest_player.draw_pointer(frame)

        if ball:
            frame = ball.draw(frame)

    frame = np.array(frame)

    # Write video
    writer.write(frame)

writer.close()

stem = os.path.basename(args.video).split(".")[0]

teams_json = os.path.join(args.out_dir, f"{stem}_teams.json")
with open(teams_json, "w") as f:
    json.dump(
        {t.name: {"color_bgr": list(t.color), "abbreviation": t.abbreviation} for t in teams},
        f, indent=2,
    )

ball_csv = os.path.join(args.out_dir, f"{stem}_ball.csv")
with open(ball_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "x", "y", "team_possession"])
    w.writerows(ball_rows)
print(f"Ball positions: {len(ball_rows)} rows saved to {ball_csv}")

if goal_detector is not None:
    goal_csv = os.path.join(args.out_dir, f"{stem}_goals.csv")
    with open(goal_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "name", "confidence", "x1", "y1", "x2", "y2", "frame_w", "frame_h"])
        w.writerows(goal_rows)
    print(f"Goal detections: {len(goal_rows)} rows saved to {goal_csv}")
else:
    print("No --goal-model given: skipping goal detections and xG (the ball CSV above is still saved).")

# Final possession summary
print("\n===== Possession =====")
for team in teams:
    print(f"{team.name}: {team.get_percentage_possession(match.duration):.0%}")
