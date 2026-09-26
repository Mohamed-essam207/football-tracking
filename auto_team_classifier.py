"""
AutoTeamClassifier: splits players into two teams automatically, with no jersey
colors to configure. Drop-in replacement for HSVClassifier in
tryolabs/soccer-video-analytics (put this file next to run.py).

How it works
------------
1. Warm-up: a few dozen frames are sampled from the video, the player detector
   is run on them, and for every player we take the median color (Lab space) of
   the torso, ignoring grass-colored pixels.
2. k-means on those colors; the two biggest clusters are the two teams. Points that are far from
   both clusters (referee, goalkeepers in a different kit, spectators) are
   trimmed out and labeled "Other" when predicting, so they are never counted
   for either team.
3. During the run, every tracked player is assigned to the nearest team by
   jersey color. InertiaClassifier (already used by run.py) then smooths the
   label per tracker ID over the last frames.
"""
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from inference.base_classifier import BaseClassifier
from inference.box import Box

OTHER = "Other"


class AutoTeamClassifier(BaseClassifier):
    def __init__(
        self,
        team_names: Sequence[str] = ("Team A", "Team B"),
        l_weight: float = 0.5,
        outlier_sigma: float = 3.0,
        min_threshold: float = 15.0,
        seed: int = 0,
    ):
        """
        team_names    : names of the two teams (must match the Team(...) names in run.py)
        l_weight      : weight of lightness vs. color in the distance (lower = less
                        sensitive to shadows / sun)
        outlier_sigma : how far (in robust std devs) from a team center a player can be
                        before being labeled "Other"
        min_threshold : lower bound for that distance, in the same units
        """
        if len(team_names) != 2:
            raise ValueError("team_names must have exactly 2 names")
        self.team_names = list(team_names)
        self.l_weight = l_weight
        self.outlier_sigma = outlier_sigma
        self.min_threshold = min_threshold
        self.seed = seed

        self.centroids: Optional[np.ndarray] = None  # (2, 3)
        self.thresholds: Optional[np.ndarray] = None  # (2,)
        self.colors_bgr: Optional[np.ndarray] = None  # (2, 3)

    # ------------------------------------------------------------------ features
    def features(self, img: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(weighted Lab feature, median BGR color) of the jersey area, or None."""
        if img is None or img.ndim != 3 or img.shape[0] < 12 or img.shape[1] < 6:
            return None

        h, w = img.shape[:2]
        crop = img[int(h * 0.15) : int(h * 0.55), int(w * 0.2) : int(w * 0.8)]
        if crop.size == 0:
            return None
        crop = np.ascontiguousarray(crop)

        pix = crop.reshape(-1, 3)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        grass = (hsv[:, 0] >= 35) & (hsv[:, 0] <= 85) & (hsv[:, 1] > 40) & (hsv[:, 2] > 40)
        keep = ~grass
        # if (almost) everything is green, the kit itself is green: use all pixels
        pix_use = pix[keep] if keep.mean() >= 0.25 else pix

        lab = cv2.cvtColor(np.ascontiguousarray(pix_use.reshape(-1, 1, 3)), cv2.COLOR_BGR2LAB)
        lab = lab.reshape(-1, 3).astype(np.float32)
        feat = np.median(lab, axis=0) * np.array([self.l_weight, 1.0, 1.0], np.float32)
        bgr = np.median(pix_use, axis=0).astype(np.float32)
        return feat, bgr

    # ---------------------------------------------------------------------- fit
    def fit_features(self, feats: List[np.ndarray], bgrs: List[np.ndarray]) -> dict:
        X = np.asarray(feats, np.float32)
        B = np.asarray(bgrs, np.float32)
        if len(X) < 8:
            raise ValueError(
                f"Only {len(X)} player crops found in the warm-up frames; "
                "need at least 8 to separate the teams."
            )

        cv2.setRNGSeed(self.seed)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
        # Over-cluster (k=4) so referee / goalkeepers get their own small clusters instead
        # of stealing one of the two team clusters; the two BIGGEST clusters are the teams.
        k = 4 if len(X) >= 60 else 2
        _, labels0, centers0 = cv2.kmeans(X, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels0.ravel(), minlength=k)
        centers = centers0[np.argsort(-counts)[:2]].copy()

        # trimmed refinement: recompute centers from inliers only
        for _ in range(3):
            d_all = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = d_all.argmin(axis=1)
            thr = np.zeros(2, np.float32)
            inlier = np.zeros(len(X), bool)
            for k in range(2):
                dk = d_all[labels == k, k]
                if len(dk) == 0:
                    raise ValueError("One of the two team clusters is empty.")
                med = np.median(dk)
                mad = np.median(np.abs(dk - med)) * 1.4826
                thr[k] = max(med + self.outlier_sigma * max(mad, 4.0), self.min_threshold)
                inlier |= (labels == k) & (d_all[:, k] <= thr[k])
            for k in range(2):
                members = X[inlier & (labels == k)]
                if len(members):
                    centers[k] = np.median(members, axis=0)

        d_all = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = d_all.argmin(axis=1)
        colors = np.zeros((2, 3), np.float32)
        sizes = []
        for k in range(2):
            members = inlier & (labels == k)
            sizes.append(int(members.sum()))
            colors[k] = np.median(B[members], axis=0) if members.any() else B[labels == k].mean(0)

        self.centroids = centers.astype(np.float32)
        self.thresholds = thr
        self.colors_bgr = colors

        separation = float(np.linalg.norm(centers[0] - centers[1]))
        return {
            "samples": int(len(X)),
            "team_sizes": sizes,
            "outliers": int(len(X) - inlier.sum()),
            "separation": separation,
            "thresholds": [float(t) for t in thr],
        }

    def fit_from_video(
        self,
        video_path: str,
        detect_fn: Callable[[np.ndarray], list],
        n_frames: int = 40,
        min_box: Tuple[int, int] = (8, 24),
        verbose: bool = True,
    ) -> dict:
        """
        Sample ~n_frames frames evenly from the video, run detect_fn(frame) (must return
        norfair Detections whose points are [[x1, y1], [x2, y2]]) and fit the two teams.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(total // max(n_frames, 1), 1)

        feats, bgrs = [], []
        i = 0
        while True:
            if i % stride == 0:
                ok, frame = cap.read()
            else:
                ok, frame = cap.grab(), None
            if not ok:
                break
            if frame is not None:
                fh, fw = frame.shape[:2]
                for det in detect_fn(frame):
                    (x1, y1), (x2, y2) = det.points[0], det.points[1]
                    x1, x2 = max(int(x1), 0), min(int(x2), fw)
                    y1, y2 = max(int(y1), 0), min(int(y2), fh)
                    if (x2 - x1) < min_box[0] or (y2 - y1) < min_box[1]:
                        continue
                    r = self.features(Box((x1, y1), (x2, y2), frame).img)
                    if r is not None:
                        feats.append(r[0])
                        bgrs.append(r[1])
            i += 1
        cap.release()

        info = self.fit_features(feats, bgrs)
        if verbose:
            print(
                f"[AutoTeamClassifier] {info['samples']} player crops from warm-up | "
                f"team sizes {info['team_sizes']} | outliers (referee/GK/other) {info['outliers']} | "
                f"team separation {info['separation']:.0f}"
            )
            if info["separation"] < 25:
                print(
                    "[AutoTeamClassifier] WARNING: the two teams' jersey colors look very similar; "
                    "the split may be unreliable."
                )
        return info

    def team_colors(self) -> List[Tuple[int, int, int]]:
        """Median jersey color per team as (B, G, R) ints, same order as team_names."""
        if self.colors_bgr is None:
            raise RuntimeError("Call fit_from_video() first.")
        return [tuple(int(v) for v in c) for c in self.colors_bgr]

    # ------------------------------------------------------------------ predict
    def _predict_one(self, img: np.ndarray) -> str:
        r = self.features(img)
        if r is None:
            return OTHER
        d = np.linalg.norm(self.centroids - r[0], axis=1)
        k = int(d.argmin())
        return self.team_names[k] if d[k] <= self.thresholds[k] else OTHER

    def predict(self, input_image: List[np.ndarray]) -> List[str]:
        if self.centroids is None:
            raise RuntimeError("Call fit_from_video() before predicting.")
        if not isinstance(input_image, list):
            input_image = [input_image]
        return [self._predict_one(img) for img in input_image]
