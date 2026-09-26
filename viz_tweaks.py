"""
viz_tweaks.py - three visual fixes for run.py (put this file next to run.py):

1. HQWriter      : writes the output with ffmpeg / H.264 at high quality. norfair's own writer
                   uses the old "mp4v" codec at a low bitrate, which is what makes the result look blocky.
2. prepare_frame : if the video is narrower than 1280 px, upscale it (bicubic) so the repo's overlays
                   fit; frames that are already >= 1280 px wide are left untouched.
3. thin_boxes    : thinner outline for the player boxes (repo default is 3 px).
"""
import shutil
import subprocess
from functools import lru_cache

import PIL.Image  # noqa: F401  (soccer.draw needs PIL.Image to be loaded first)
import PIL.ImageDraw  # noqa: F401
import PIL.ImageFont  # noqa: F401
import cv2
import numpy as np

from soccer.draw import Draw

# ---- settings you may want to tweak ----------------------------------------
BOX_THICKNESS = 2   # player box line width in px (repo default: 3, use 1 for very thin)
CRF = 16            # H.264 quality: lower = better + bigger file (14 ~ near lossless, 18 = good)
PRESET = "slow"     # slower = a bit smaller file at the same quality
MIN_WIDTH = 1280    # upscale videos narrower than this
GOAL_COLOR = (255, 220, 0)  # goal boxes; BGR order, shows up as cyan in the video
GOAL_WIDTH = 2
GOAL_FONT = "fonts/Gidole-Regular.ttf"  # comes with the repo; run from the repo root
# ------------------------------------------------------------------------------


def thin_boxes(thickness: int = BOX_THICKNESS):
    """Make Draw.draw_bounding_box (used for every player box) use a thinner line."""
    if getattr(Draw.draw_bounding_box, "_thin", False):
        return
    original = Draw.draw_bounding_box

    def thin(img, rectangle, color, thickness_arg=None, **kwargs):
        return original(img=img, rectangle=rectangle, color=color, thickness=thickness)

    thin._thin = True
    Draw.draw_bounding_box = staticmethod(thin)


def prepare_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w >= MIN_WIDTH:
        return frame
    new_w = MIN_WIDTH
    new_h = int(round(h * MIN_WIDTH / w / 2)) * 2  # keep the height even (H.264 needs it)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


class HQWriter:
    def __init__(self, video, fps: float, crf: int = CRF, preset: str = PRESET):
        """video: the norfair Video (used only for the output file name / as fallback)."""
        self.video = video
        self.fps = fps if fps and fps > 0 else 30.0
        self.crf = crf
        self.preset = preset
        self.path = video.get_output_file_path()
        self.proc = None
        self.use_ffmpeg = shutil.which("ffmpeg") is not None
        if not self.use_ffmpeg:
            print("[HQWriter] ffmpeg not found: falling back to norfair's writer (lower quality).")

    def write(self, frame: np.ndarray):
        if not self.use_ffmpeg:
            self.video.write(frame)
            return
        if self.proc is None:
            h, w = frame.shape[:2]
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{self.fps}",
                "-i", "-",
                "-an",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                self.path,
            ]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            raise RuntimeError("ffmpeg stopped unexpectedly while writing the video.")

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
            print(f"Output video file saved to: {self.path}")
            self.proc = None


@lru_cache(maxsize=4)
def _goal_font(size: int):
    try:
        return PIL.ImageFont.truetype(GOAL_FONT, size=size)
    except OSError:
        return PIL.ImageFont.load_default()


def draw_goal_boxes(frame, goal_df, size: int = 15):
    """Draw the detections of the goal model (thin cyan box + name and confidence)."""
    if goal_df is None or len(goal_df) == 0:
        return frame
    draw = PIL.ImageDraw.Draw(frame)
    font = _goal_font(size)
    for r in goal_df.to_dict("records"):
        x1, y1, x2, y2 = r["xmin"], r["ymin"], r["xmax"], r["ymax"]
        draw.rectangle([x1, y1, x2, y2], outline=GOAL_COLOR, width=GOAL_WIDTH)
        draw.text((x1 + 3, max(y1 - size - 4, 0)), f'{r["name"]} {r["confidence"]:.2f}', fill=GOAL_COLOR, font=font)
    return frame
