
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", default="0")

    args = parser.parse_args()

    model = YOLO(args.weights)

    print("Starting independent validation...")

    metrics = model.val(
        data=args.data,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        workers=0,
        max_det=args.max_det,
        plots=False,
        save_json=False
    )

    print("\nValidation Results")
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)


if __name__ == "__main__":
    main()
