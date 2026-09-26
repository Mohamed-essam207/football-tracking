"""
xg_overlay.py - draws the shots + a running xG total onto the video run.py already produced
(put next to run.py). Runs AFTER compute_xg.py.

    python xg_overlay.py --video outputs/match_out.mp4 --shots outputs/match_ball_shots.csv \
        --teams outputs/match_teams.json --out outputs/match_out_xg.mp4

For every shot: a circle appears at the shot location for ~1 second with a small
"Shot | xG 0.12" label. A compact panel in the bottom-right accumulates each team's total xG as
shots happen, so it fills in over the course of the video, next to the video's own possession
board. See compute_xg.py's docstring for what the xG numbers do and don't account for.
"""
import argparse
import csv
import json
from functools import lru_cache

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from viz_tweaks import HQWriter

FONT_PATH = "fonts/Gidole-Regular.ttf"
MARKER_COLOR = (0, 210, 255)  # BGR, like the rest of this codebase -> amber in the video
MARKER_SECONDS = 1.0
PANEL_W, PANEL_H = 300, 92  # slightly shorter than the possession board so it doesn't overlap on short videos
PANEL_MARGIN = 30
PANEL_BG = (20, 20, 24, 205)
PANEL_BORDER = (255, 255, 255, 45)
TEXT_MAIN = (255, 255, 255, 255)
TEXT_DIM = (176, 176, 184, 255)


@lru_cache(maxsize=8)
def _font(size):
    try:
        return PIL.ImageFont.truetype(FONT_PATH, size=size)
    except OSError:
        return PIL.ImageFont.load_default()


def _text(draw, xy, text, size, fill, anchor="l"):
    font = _font(size)
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    w, h = r - l, b - t
    x, y = xy
    if anchor == "m":
        x -= w / 2
    elif anchor == "r":
        x -= w
    draw.text((x - l, y - h / 2 - t), text, font=font, fill=fill)


def _rgb(color):
    return tuple(int(c) for c in color[:3])


def load_shots(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["frame"] = int(r["frame"])
        r["x"], r["y"] = float(r["x"]), float(r["y"])
        r["xg"] = float(r["xg"])
    rows.sort(key=lambda r: r["frame"])
    return rows


def load_teams(path):
    try:
        with open(path) as f:
            data = json.load(f)
        return {name: _rgb(t["color_bgr"]) + (255,) for name, t in data.items()}
    except (OSError, KeyError):
        return {}


def draw_marker(frame, shot, age_frames, total_frames):
    fade = max(0.0, 1.0 - age_frames / max(total_frames, 1))
    r = 14 + int(10 * (age_frames / max(total_frames, 1)))  # grows slightly as it fades
    x, y = shot["x"], shot["y"]
    overlay = PIL.Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = PIL.ImageDraw.Draw(overlay)
    alpha = int(255 * fade)
    d.ellipse([x - r, y - r, x + r, y + r], outline=MARKER_COLOR + (alpha,), width=3)
    label = f'Shot  \u00b7  xG {shot["xg"]:.2f}'
    lx, ly = x, max(y - r - 14, 16)
    bbox = d.textbbox((0, 0), label, font=_font(16))
    pad = 5
    d.rounded_rectangle(
        [lx - (bbox[2] - bbox[0]) / 2 - pad, ly - (bbox[3] - bbox[1]) / 2 - pad,
         lx + (bbox[2] - bbox[0]) / 2 + pad, ly + (bbox[3] - bbox[1]) / 2 + pad],
        radius=6, fill=(20, 20, 24, min(alpha, 205)),
    )
    _text(d, (lx, ly), label, 16, MARKER_COLOR + (alpha,), "m")
    return PIL.Image.alpha_composite(frame.convert("RGBA"), overlay).convert(frame.mode)


def draw_xg_panel(frame, totals, team_colors, origin=None):
    W, H = frame.size
    s = W / 1280.0
    pw, ph = int(PANEL_W * s), int(PANEL_H * s)
    margin = int(PANEL_MARGIN * s)
    # sits just below where the possession board goes (top-right), left of the frame edge
    x0, y0 = origin if origin else (W - pw - margin, margin + int(112 * s))

    img = PIL.Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    d = PIL.ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, pw - 1, ph - 1], radius=int(14 * s), fill=PANEL_BG, outline=PANEL_BORDER, width=1)
    _text(d, (pw / 2, 16 * s), "TOTAL xG", int(11 * s), TEXT_DIM, "m")

    names = list(totals.keys())
    row_y = 46 * s
    if len(names) == 0:
        _text(d, (pw / 2, row_y), "no shots yet", int(13 * s), TEXT_DIM, "m")
    else:
        col_w = pw / len(names)
        for i, name in enumerate(names):
            cx = col_w * (i + 0.5)
            color = team_colors.get(name, TEXT_MAIN)
            _text(d, (cx, row_y), name, int(12 * s), color, "m")
            _text(d, (cx, row_y + 26 * s), f"{totals[name]:.2f}", int(20 * s), TEXT_MAIN, "m")

    box = (x0, y0, x0 + pw, y0 + ph)
    region = frame.crop(box).convert("RGBA")
    region = PIL.Image.alpha_composite(region, img)
    frame.paste(region.convert(frame.mode), (x0, y0))
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="The tracking+possession video from run.py (*_out.mp4)")
    ap.add_argument("--shots", required=True, help="*_ball_shots.csv from compute_xg.py")
    ap.add_argument("--teams", default=None, help="*_teams.json from run.py (for the real jersey colors)")
    ap.add_argument("--out", default=None, help="Output path (default: <video>_xg.mp4)")
    args = ap.parse_args()

    import cv2  # local import: keeps this file's own deps explicit even if cv2 patches waitKey elsewhere

    shots = load_shots(args.shots)
    team_colors = load_teams(args.teams) if args.teams else {}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    marker_frames = max(int(round(MARKER_SECONDS * fps)), 1)

    out_path = args.out or args.video.rsplit(".", 1)[0] + "_xg.mp4"

    class _P:  # tiny stand-in so HQWriter can reuse its ffmpeg pipeline here
        def get_output_file_path(self):
            return out_path

    writer = HQWriter(_P(), fps)

    shots_by_frame = {}
    for s in shots:
        shots_by_frame.setdefault(s["frame"], []).append(s)

    totals = {}
    active = []  # shots currently being drawn: (shot, start_frame)
    i = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        for s in shots_by_frame.get(i, []):
            team = s["team"] or "Unknown"
            totals[team] = totals.get(team, 0.0) + s["xg"]
            active.append((s, i))
        active = [(s, f0) for s, f0 in active if i - f0 < marker_frames]

        frame = PIL.Image.fromarray(frame_bgr)
        for s, f0 in active:
            frame = draw_marker(frame, s, i - f0, marker_frames)
        frame = draw_xg_panel(frame, totals, team_colors)

        writer.write(np.array(frame))
        i += 1
        if i % 100 == 0:
            print(f"{i}/{n_frames} frames")

    cap.release()
    writer.close()
    print(f"\nFinal totals: {', '.join(f'{k}: {v:.2f}' for k, v in totals.items()) or '(no shots)'}")


if __name__ == "__main__":
    main()
