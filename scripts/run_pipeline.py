import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import argparse, cv2
from pathlib import Path
from src.perception.pipeline import PerceptionPipeline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Video path or camera index")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--output", default="outputs/perception.mp4")
    p.add_argument("--depth", action="store_true", help="Enable MiDaS (downloads model on first run)")
    p.add_argument("--display", action="store_true")
    a = p.parse_args()
    source = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): raise SystemExit(f"Cannot open source: {a.source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1: fps = 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    pipe = PerceptionPipeline(a.weights, enable_depth=a.depth)
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            out, _ = pipe.process(frame)
            writer.write(out)
            if a.display:
                cv2.imshow("Perception Stack - press q to quit", out)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
    finally:
        cap.release(); writer.release(); cv2.destroyAllWindows()
    print("Saved:", a.output)
if __name__ == "__main__": main()
