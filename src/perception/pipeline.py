import cv2
import numpy as np
from ultralytics import YOLO
from .lane import detect_lanes

class PerceptionPipeline:
    def __init__(self, weights="yolov8n.pt", conf=0.25, enable_depth=False):
        self.detector = YOLO(weights)
        self.conf = conf
        self.depth = None
        if enable_depth:
            from .depth import MiDaSDepth
            self.depth = MiDaSDepth()

    def process(self, frame):
        result = self.detector.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=self.conf, verbose=False
        )[0]
        annotated = result.plot()
        annotated = detect_lanes(annotated)
        if self.depth is not None:
            depth = self.depth.predict(frame)
            heat = cv2.applyColorMap((depth*255).astype("uint8"), cv2.COLORMAP_MAGMA)
            heat = cv2.resize(heat, (annotated.shape[1]//4, annotated.shape[0]//4))
            h, w = heat.shape[:2]
            annotated[10:10+h, 10:10+w] = heat
            cv2.putText(annotated, "MiDaS relative depth (not metres)", (10, h+28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return annotated, result
