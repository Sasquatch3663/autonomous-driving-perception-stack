import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import argparse, cv2
from src.perception.lane import detect_lanes
p=argparse.ArgumentParser(); p.add_argument("--source",required=True); p.add_argument("--output",default="outputs/lane.jpg")
a=p.parse_args()
im=cv2.imread(a.source)
if im is None: raise SystemExit("Could not read image")
out=detect_lanes(im); cv2.imwrite(a.output,out); print("Saved:",a.output)
