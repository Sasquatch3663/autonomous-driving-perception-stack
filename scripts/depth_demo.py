import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import argparse, cv2
from src.perception.depth import MiDaSDepth
p=argparse.ArgumentParser(); p.add_argument("--source",required=True); p.add_argument("--output",default="outputs/depth.jpg")
a=p.parse_args()
im=cv2.imread(a.source)
if im is None: raise SystemExit("Could not read image")
depth=MiDaSDepth().predict(im)
heat=cv2.applyColorMap((depth*255).astype("uint8"),cv2.COLORMAP_MAGMA)
cv2.imwrite(a.output,heat); print("Saved relative-depth visualization:",a.output)
