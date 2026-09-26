"""
UltralyticsDetector: same interface as the repo's YoloV5 detector
(predict(frame) -> DataFrame with xmin, ymin, xmax, ymax, confidence, class, name),
but loads a model trained with the `ultralytics` package (YOLOv8 / YOLO11).

The repo's YoloV5 class can only load YOLOv5-format weights, so it fails on a
YOLOv8 model with: "'Detect' object has no attribute 'grid'".
"""
import numpy as np
import pandas as pd

from inference.base_detector import BaseDetector

COLUMNS = ["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]


class UltralyticsDetector(BaseDetector):
    def __init__(self, model_path: str, conf: float = 0.25, imgsz: int = 640):
        """
        model_path : path to the .pt trained with ultralytics
        conf       : minimum confidence returned by the model (run_utils filters again at 0.3)
        imgsz      : inference size; raise it (e.g. 960 / 1280) if the ball is small in your video
        """
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        print(f"[UltralyticsDetector] loaded {model_path} | classes: {self.model.names}")

    def predict(self, input_image: np.ndarray) -> pd.DataFrame:
        # numpy frames are treated as BGR (cv2 convention), which is what the video gives us
        result = self.model.predict(
            input_image, conf=self.conf, imgsz=self.imgsz, verbose=False
        )[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return pd.DataFrame(columns=COLUMNS)

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        return pd.DataFrame(
            {
                "xmin": xyxy[:, 0],
                "ymin": xyxy[:, 1],
                "xmax": xyxy[:, 2],
                "ymax": xyxy[:, 3],
                "confidence": conf,
                "class": cls,
                "name": [result.names[int(c)] for c in cls],
            }
        )
