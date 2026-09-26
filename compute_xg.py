"""
compute_xg.py - shot detection and a simple geometric xG, from the CSVs run.py already saves
(no shot-outcome dataset needed, and nothing to train).

    python compute_xg.py --ball outputs/match_ball.csv --goals outputs/match_goals.csv --fps 30

What it does
------------
1. Shot detection: looks at the ball's frame-to-frame speed and direction. A "shot" is a burst
   of frames where the ball moves fast and roughly straight at a goal. Nearby bursts are merged
   into one shot; the shot's location is the ball position where the burst starts (i.e. roughly
   where it was kicked from), not where it ends up.
2. xG: uses the goal box the goal model already found (its pixel width stands in for the real
   goal width, 7.32 m, so pixel distances near the goal can be turned into meters) to get the
   shot's distance and shooting angle to goal, then a hand-set logistic curve turns those into a
   probability.

IMPORTANT — this is a heuristic, not a statistical model: the curve's coefficients were picked by
hand to look reasonable (closer + straighter-on = higher), not fitted to real shot outcomes. Two
approximations to keep in mind: the meters-per-pixel scale is derived from the goal box, which is
least accurate for shots taken from far away or from a sharp angle; and it doesn't know about
defenders, the goalkeeper's position, or whether the shot was blocked. Treat the numbers as
"relative shot quality on this clip", not literal goal probabilities. If you later collect real
shots with known outcomes (goal / no goal), that data can replace this formula with a fitted
logistic regression, which is what real xG models use.
"""
import argparse
import csv
import math

GOAL_WIDTH_M = 7.32  # official width of a goal mouth, post to post


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ball", required=True, help="<video>_ball.csv from run.py")
    p.add_argument("--goals", required=True, help="<video>_goals.csv from run.py")
    p.add_argument("--fps", type=float, required=True, help="Video fps (printed by run.py / ffprobe)")
    p.add_argument("--out", default=None, help="Output CSV path (default: <ball csv>_shots.csv)")
    p.add_argument("--min-speed-percentile", type=float, default=90,
                    help="A frame counts as fast if its ball speed is above this percentile of all speeds")
    p.add_argument("--min-alignment", type=float, default=0.5,
                    help="cos(angle) between the ball's velocity and the ball->goal direction; "
                         "1.0 = moving straight at goal, 0.0 = moving sideways to it")
    p.add_argument("--merge-gap", type=int, default=8, help="Merge shot frames closer than this (frames)")
    p.add_argument("--lookback", type=int, default=6,
                    help="Frames to look back from the speed peak for the shot's origin/team")
    p.add_argument("--after-frames", type=int, default=20,
                    help="Frames after the shot to check whether the ball reached the goal box")
    p.add_argument("--max-distance-m", type=float, default=45,
                    help="Discard candidates farther than this from goal: a fast, roughly-toward-goal "
                         "ball movement from way outside the pitch is a pass caught by the alignment "
                         "check, not a shot, since real shots rarely come from this far")
    # hand-picked logistic coefficients: xG = sigmoid(a + b*distance_m + c*angle_deg)
    p.add_argument("--coef-a", type=float, default=-1.0)
    p.add_argument("--coef-dist", type=float, default=-0.10)
    p.add_argument("--coef-angle", type=float, default=0.05)
    return p.parse_args()


def read_ball_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            x = float(r["x"]) if r["x"] not in ("", None) else None
            y = float(r["y"]) if r["y"] not in ("", None) else None
            rows.append({"frame": int(r["frame"]), "x": x, "y": y, "team": r["team_possession"] or None})
    rows.sort(key=lambda r: r["frame"])
    return rows


def read_goal_csv(path):
    by_frame = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            frame = int(r["frame"])
            row = {k: float(r[k]) for k in ("confidence", "x1", "y1", "x2", "y2")}
            if frame not in by_frame or row["confidence"] > by_frame[frame]["confidence"]:
                by_frame[frame] = row
    return by_frame


def fill_goal_per_frame(goal_by_frame, n_frames):
    """Forward/backward-fill the goal box so every frame has one (camera moves slowly frame to frame)."""
    known = sorted(goal_by_frame)
    if not known:
        return [None] * n_frames
    filled, last = [], None
    ki = 0
    for f in range(n_frames):
        while ki < len(known) and known[ki] <= f:
            last = goal_by_frame[known[ki]]
            ki += 1
        filled.append(last)
    # back-fill the very first frames, which had no goal detection yet
    first_known = goal_by_frame[known[0]]
    for f in range(len(filled)):
        if filled[f] is None:
            filled[f] = first_known
        else:
            break
    return filled


def percentile(values, p):
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * p / 100
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    args = parse_args()
    ball = read_ball_csv(args.ball)
    n = ball[-1]["frame"] + 1 if ball else 0
    goal_by_frame = read_goal_csv(args.goals)
    goal_per_frame = fill_goal_per_frame(goal_by_frame, n)

    pos = {r["frame"]: (r["x"], r["y"]) for r in ball if r["x"] is not None}
    team = {r["frame"]: r["team"] for r in ball}

    # velocity + speed, only between frames that are close enough together (small tracking gaps are ok)
    speed = {}
    for f in range(1, n):
        if f in pos and (f - 1) in pos:
            x0, y0 = pos[f - 1]
            x1, y1 = pos[f]
            speed[f] = (x1 - x0, y1 - y0, math.hypot(x1 - x0, y1 - y0) * args.fps)

    if not speed:
        print("No consecutive ball detections found: cannot detect shots.")
        return

    speed_threshold = percentile([s[2] for s in speed.values()], args.min_speed_percentile)

    candidates = []
    for f, (vx, vy, sp) in speed.items():
        if sp < speed_threshold or f not in pos:
            continue
        goal = goal_per_frame[f]
        if goal is None:
            continue
        gx, gy = (goal["x1"] + goal["x2"]) / 2, (goal["y1"] + goal["y2"]) / 2
        bx, by = pos[f]
        to_goal = (gx - bx, gy - by)
        norm_v = math.hypot(vx, vy)
        norm_g = math.hypot(*to_goal)
        if norm_v == 0 or norm_g == 0:
            continue
        alignment = (vx * to_goal[0] + vy * to_goal[1]) / (norm_v * norm_g)
        if alignment >= args.min_alignment:
            candidates.append(f)

    if not candidates:
        print(f"No shot-like ball movement found (speed > {speed_threshold:.0f} px/s, "
              f"heading at the goal). Try lowering --min-speed-percentile or --min-alignment.")
        return

    # merge nearby candidate frames into shot events
    candidates.sort()
    events = [[candidates[0]]]
    for f in candidates[1:]:
        if f - events[-1][-1] <= args.merge_gap:
            events[-1].append(f)
        else:
            events.append([f])

    shots = []
    for ev in events:
        peak = max(ev, key=lambda f: speed[f][2])
        origin_f = max(peak - args.lookback, min(ev))
        while origin_f not in pos and origin_f < peak:
            origin_f += 1
        if origin_f not in pos:
            continue
        bx, by = pos[origin_f]

        shot_team = team.get(origin_f) or next(
            (team[f] for f in range(origin_f, max(origin_f - args.lookback, 0) - 1, -1) if team.get(f)), None
        )

        goal = goal_per_frame[peak]
        x1, y1, x2, y2 = goal["x1"], goal["y1"], goal["x2"], goal["y2"]
        goal_px_width = max(x2 - x1, 1.0)
        scale = GOAL_WIDTH_M / goal_px_width  # meters per pixel, calibrated off the goal box

        gcx, gcy = (x1 + x2) / 2, (y1 + y2) / 2
        dist_px = math.hypot(gcx - bx, gcy - by)
        distance_m = dist_px * scale

        gy_mid = (y1 + y2) / 2
        a = (x1 - bx, gy_mid - by)
        b = (x2 - bx, gy_mid - by)
        na, nb = math.hypot(*a), math.hypot(*b)
        angle_deg = 0.0
        if na > 0 and nb > 0:
            cos_a = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb)))
            angle_deg = math.degrees(math.acos(cos_a))

        z = args.coef_a + args.coef_dist * distance_m + args.coef_angle * angle_deg
        xg = 1 / (1 + math.exp(-z))

        if distance_m > args.max_distance_m:
            continue

        reached_goal = any(
            f in pos and x1 <= pos[f][0] <= x2 and y1 <= pos[f][1] <= y2
            for f in range(peak, min(peak + args.after_frames, n))
        )

        shots.append(
            dict(frame=origin_f, time_s=round(origin_f / args.fps, 1), team=shot_team or "",
                 x=round(bx, 1), y=round(by, 1), distance_m=round(distance_m, 1),
                 angle_deg=round(angle_deg, 1), xg=round(xg, 3),
                 reached_goal_box=reached_goal)
        )

    out_path = args.out or args.ball.replace(".csv", "") + "_shots.csv"
    if out_path == args.ball:
        out_path = args.ball[:-4] + "_shots.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "time_s", "team", "x", "y", "distance_m",
                                           "angle_deg", "xg", "reached_goal_box"])
        w.writeheader()
        w.writerows(shots)

    print(f"{len(shots)} shot(s) detected -> {out_path}\n")
    totals = {}
    for s in shots:
        t = s["team"] or "Unknown"
        totals.setdefault(t, {"shots": 0, "xg": 0.0, "reached": 0})
        totals[t]["shots"] += 1
        totals[t]["xg"] += s["xg"]
        totals[t]["reached"] += s["reached_goal_box"]
    print("===== xG summary (heuristic, see the file's docstring) =====")
    for t, v in totals.items():
        print(f"{t}: {v['shots']} shot(s), xG {v['xg']:.2f}, ball reached the goal box on {v['reached']}")


if __name__ == "__main__":
    main()
