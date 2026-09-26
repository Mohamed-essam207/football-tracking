"""
Custom possession board for tryolabs/soccer-video-analytics (put next to run.py).

Same information as the repo's board (team names, possession time, possession %),
different look: a compact dark panel at the top-right with team chips, big percentages,
possession time in the middle and a split bar. A small amber dot marks the team that
currently has the ball.

Every visual choice is a constant at the top of the file, so it is easy to tweak.
Colors here are written in BGR order because the repo hands PIL a BGR frame (that is why
ACCENT = (0, 190, 255) shows up as amber in the video).
"""
from functools import lru_cache

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

FONT_PATH = "fonts/Gidole-Regular.ttf"  # comes with the repo; run from the repo root

# ---- look & feel (base size: 1280 px wide frame, everything scales with the frame width) ----
PANEL_SIZE = (430, 104)
PANEL_RADIUS = 16
PASSES_PANEL_SIZE = (430, 74)
PASSES_GAP = 14  # gap between the bottom of the possession board and the top of the passes board
PANEL_BG = (20, 20, 24, 205)  # RGBA, last number = opacity
PANEL_BORDER = (255, 255, 255, 45)
TRACK = (60, 60, 66, 255)
TEXT_MAIN = (255, 255, 255, 255)
TEXT_DIM = (176, 176, 184, 255)
ACCENT = (0, 190, 255, 255)  # "has the ball" dot (BGR -> amber)
TITLE = "BALL POSSESSION"
MARGIN = 30  # distance from the top / right edge
SS = 2  # supersampling, for smooth edges


@lru_cache(maxsize=32)
def _font(size: int):
    try:
        return PIL.ImageFont.truetype(FONT_PATH, size=size)
    except OSError:
        try:
            return PIL.ImageFont.load_default(size=size)
        except TypeError:
            return PIL.ImageFont.load_default()


def _text(draw, xy, text, size, fill, anchor="l"):
    """Draw text with its vertical middle at xy[1]; anchor = l / m / r for the x position."""
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


def draw_possession_board(frame: PIL.Image.Image, match, origin=None, scale=None):
    """
    frame  : PIL image (the same one run.py draws on)
    match  : soccer.Match
    origin : (x, y) of the panel's top-left corner; default = top-right
    scale  : override the automatic scale (frame_width / 1280)
    """
    W, H = frame.size
    s = scale if scale else W / 1280.0
    pw, ph = int(round(PANEL_SIZE[0] * s)), int(round(PANEL_SIZE[1] * s))
    margin = int(round(MARGIN * s))
    x0, y0 = origin if origin else (W - pw - margin, margin)

    k = s * SS

    def u(v):  # base units -> supersampled pixels
        return int(round(v * k))

    home, away = match.home, match.away
    home_c, away_c = _rgb(home.board_color) + (255,), _rgb(away.board_color) + (255,)

    # possession split
    if match.duration > 0:
        ratio = home.get_percentage_possession(match.duration)
    else:
        ratio = 0.5
    home_pct = int(round(ratio * 100))
    away_pct = 100 - home_pct
    ratio = min(max(ratio, 0.04), 0.96)  # keep both bar segments visible

    W0, H0 = PANEL_SIZE
    img = PIL.Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    d = PIL.ImageDraw.Draw(img)

    # panel
    d.rounded_rectangle(
        [0, 0, pw * SS - 1, ph * SS - 1],
        radius=u(PANEL_RADIUS),
        fill=PANEL_BG,
        outline=PANEL_BORDER,
        width=SS,
    )

    # team chips + title
    chip_w, chip_h, chip_y = 66, 26, 12
    for team, color, x in ((home, home_c, 16), (away, away_c, W0 - 16 - chip_w)):
        d.rounded_rectangle(
            [u(x), u(chip_y), u(x + chip_w), u(chip_y + chip_h)],
            radius=u(chip_h / 2),
            fill=color,
        )
        _text(d, (u(x + chip_w / 2), u(chip_y + chip_h / 2)), team.abbreviation,
              u(15), _rgb(team.text_color) + (255,), "m")
    _text(d, (u(W0 / 2), u(chip_y + chip_h / 2)), TITLE, u(12), TEXT_DIM, "m")

    # "has the ball" dot
    dot_y, dot_r = chip_y + chip_h / 2, 4
    holder = match.team_possession
    if holder is not None:
        dot_x = 16 + chip_w + 12 if holder is home else W0 - 16 - chip_w - 12
        d.ellipse([u(dot_x - dot_r), u(dot_y - dot_r), u(dot_x + dot_r), u(dot_y + dot_r)], fill=ACCENT)

    # percentages + possession time
    row_y = 55
    _text(d, (u(16), u(row_y)), f"{home_pct}%", u(32), TEXT_MAIN, "l")
    _text(d, (u(W0 - 16), u(row_y)), f"{away_pct}%", u(32), TEXT_MAIN, "r")
    times = f"{home.get_time_possession(match.fps)}  -  {away.get_time_possession(match.fps)}"
    _text(d, (u(W0 / 2), u(row_y)), times, u(14), TEXT_DIM, "m")

    # split bar
    bx0, bx1, by0, by1 = 16, W0 - 16, 80, 92
    split = bx0 + ratio * (bx1 - bx0)
    r = (by1 - by0) / 2
    d.rounded_rectangle([u(bx0), u(by0), u(bx1), u(by1)], radius=u(r), fill=TRACK)
    d.rounded_rectangle([u(bx0), u(by0), u(split - 2), u(by1)], radius=u(r), fill=home_c)
    d.rounded_rectangle([u(split + 2), u(by0), u(bx1), u(by1)], radius=u(r), fill=away_c)

    overlay = img.resize((pw, ph), PIL.Image.LANCZOS)

    box = (x0, y0, x0 + pw, y0 + ph)
    region = frame.crop(box).convert("RGBA")
    region = PIL.Image.alpha_composite(region, overlay)
    frame.paste(region.convert(frame.mode), (x0, y0))
    return frame


def draw_passes_board(frame: PIL.Image.Image, match, origin=None, scale=None):
    """
    Small panel with each team's pass count, meant to sit directly under draw_possession_board.
    Pass counting itself is the repo's own logic (soccer/pass_event.py): a pass is counted
    whenever the ball moves from one player to a different player on the same team.
    """
    W, H = frame.size
    s = scale if scale else W / 1280.0
    pw, ph = int(round(PASSES_PANEL_SIZE[0] * s)), int(round(PASSES_PANEL_SIZE[1] * s))
    margin = int(round(MARGIN * s))
    if origin:
        x0, y0 = origin
    else:
        pos_h = int(round(PANEL_SIZE[1] * s))
        x0 = W - pw - margin
        y0 = margin + pos_h + int(round(PASSES_GAP * s))

    k = s * SS

    def u(v):
        return int(round(v * k))

    home, away = match.home, match.away
    home_c, away_c = _rgb(home.board_color) + (255,), _rgb(away.board_color) + (255,)
    home_n, away_n = len(home.passes), len(away.passes)
    total = home_n + away_n
    ratio = home_n / total if total else 0.5
    ratio = min(max(ratio, 0.04), 0.96) if total else 0.5

    W0, H0 = PASSES_PANEL_SIZE
    img = PIL.Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    d = PIL.ImageDraw.Draw(img)

    d.rounded_rectangle(
        [0, 0, pw * SS - 1, ph * SS - 1],
        radius=u(PANEL_RADIUS), fill=PANEL_BG, outline=PANEL_BORDER, width=SS,
    )

    row_y = 30
    _text(d, (u(16), u(row_y)), str(home_n), u(26), home_c, "l")
    _text(d, (u(W0 - 16), u(row_y)), str(away_n), u(26), away_c, "r")
    _text(d, (u(W0 / 2), u(row_y - 12)), "PASSES", u(11), TEXT_DIM, "m")
    _text(d, (u(W0 / 2), u(row_y + 12)), f"{home.abbreviation}  -  {away.abbreviation}", u(11), TEXT_DIM, "m")

    bx0, bx1, by0, by1 = 16, W0 - 16, 58, 66
    r = (by1 - by0) / 2
    d.rounded_rectangle([u(bx0), u(by0), u(bx1), u(by1)], radius=u(r), fill=TRACK)
    if total:
        split = bx0 + ratio * (bx1 - bx0)
        d.rounded_rectangle([u(bx0), u(by0), u(split - 2), u(by1)], radius=u(r), fill=home_c)
        d.rounded_rectangle([u(split + 2), u(by0), u(bx1), u(by1)], radius=u(r), fill=away_c)

    overlay = img.resize((pw, ph), PIL.Image.LANCZOS)
    box = (x0, y0, x0 + pw, y0 + ph)
    region = frame.crop(box).convert("RGBA")
    region = PIL.Image.alpha_composite(region, overlay)
    frame.paste(region.convert(frame.mode), (x0, y0))
    return frame
