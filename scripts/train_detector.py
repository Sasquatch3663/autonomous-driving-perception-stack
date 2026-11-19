
import argparse
import torch
from ultralytics import YOLO


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train YOLO on BDD100K using GPU or CPU"
    )

    # Dataset and model
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolov8n.pt")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)

    # Dataset fraction
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of training dataset to use (0.0 to 1.0)"
    )

    # Optional caching
    parser.add_argument(
        "--cache",
        choices=["none", "disk", "ram"],
        default="none",
        help="Dataset caching method"
    )

    # Optional experiment tracking
    parser.add_argument("--wandb", action="store_true")

    return parser.parse_args()


def check_hardware(device):

    print("\n========== HARDWARE INFORMATION ==========")

    cuda_available = torch.cuda.is_available()

    if cuda_available:
        print("CUDA available: True")
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)

        gpu_memory = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        print("GPU memory:", round(gpu_memory, 2), "GB")

    else:
        print("CUDA available: False")
        print("No CUDA-compatible GPU detected.")

    print("===========================================\n")

    if device == "auto":
        selected_device = 0 if cuda_available else "cpu"

    elif device == "0":
        if not cuda_available:
            raise RuntimeError(
                "GPU requested but CUDA is unavailable. "
                "Install CUDA-enabled PyTorch."
            )

        selected_device = 0

    elif device == "cpu":
        selected_device = "cpu"

    else:
        raise ValueError(
            "Use --device 0, --device cpu, or --device auto"
        )

    print("Selected training device:", selected_device)

    return selected_device


def train_model(args, device):

    # Optional Weights & Biases
    if args.wandb:
        try:
            import wandb
            wandb.login()

        except ImportError:
            raise RuntimeError(
                "W&B is not installed. Run: pip install wandb"
            )

    # Validate dataset fraction
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError(
            "--fraction must be greater than 0 and at most 1.0"
        )

    # Load pretrained YOLO model
    print("\nLoading YOLO model:", args.model)

    model = YOLO(args.model)

    # Convert cache option
    cache_option = False if args.cache == "none" else args.cache

    print("\n========== TRAINING CONFIGURATION ==========")

    print("Dataset:", args.data)
    print("Model:", args.model)
    print("Epochs:", args.epochs)
    print("Image size:", args.imgsz)
    print("Batch size:", args.batch)
    print("Workers:", args.workers)
    print("Device:", device)
    print("Dataset fraction:", args.fraction)
    print("Cache:", args.cache)
    print("Validation during training: Disabled")
    print("AMP: Enabled (Ultralytics may disable it automatically)")

    print("============================================\n")

    # Train model
    results = model.train(

        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,

        device=device,
        workers=args.workers,

        # Dataset optimization
        fraction=args.fraction,
        cache=cache_option,

        # GPU settings
        amp=True,

        # Disable regular validation
        # Final validation can still run in Ultralytics
        val=False,

        # Checkpoint saving
        save=True,
        save_period=1,

        # Reduce additional overhead
        plots=False,
        max_det=100,

        # Training configuration
        project="runs/detect",
        name="bdd100k_fast",
        exist_ok=True,

        patience=15,
        verbose=True
    )

    print("\n========== TRAINING COMPLETED ==========")

    print("Training results:", results)

    print("\nCheckpoints are stored in:")

    print("runs/detect/bdd100k_fast/weights/")

    print("\nExpected checkpoint:")

    print("last.pt")

    print("Note: best.pt may not be generated because validation is disabled.")

    print("========================================")

    return model


def main():

    args = parse_arguments()

    try:

        device = check_hardware(args.device)

        model = train_model(args, device)

        print("\nYOLO training completed successfully!")

    except KeyboardInterrupt:

        print("\nTraining interrupted by user.")

    except Exception as error:

        print("\nTRAINING ERROR:")
        print(error)

        raise


if __name__ == "__main__":
    main()
