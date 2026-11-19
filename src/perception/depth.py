import cv2
import numpy as np
import torch

class MiDaSDepth:
    """MiDaS relative-depth estimator. Output is not calibrated metres."""
    def __init__(self, model_type="MiDaS_small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = (transforms.small_transform if "small" in model_type
                          else transforms.default_transform)

    @torch.inference_mode()
    def predict(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)
        pred = self.model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic",
            align_corners=False
        ).squeeze()
        depth = pred.cpu().numpy()
        lo, hi = np.percentile(depth, [2, 98])
        normalized = np.clip((depth-lo)/(hi-lo+1e-8), 0, 1)
        return normalized
