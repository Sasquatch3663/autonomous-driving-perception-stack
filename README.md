# Autonomous Driving Perception Stack

A modular research/demo pipeline for camera-based autonomous-driving perception:
- YOLO object detection (Ultralytics)
- ByteTrack multi-object tracking
- Classical OpenCV lane detection
- Optional semantic lane segmentation interface (SegFormer via Hugging Face)
- MiDaS monocular relative-depth estimation
- CARLA camera capture and optional ROS 2 image publisher
- ONNX export and latency/FPS benchmarking
- Optional Weights & Biases experiment logging

> This is a runnable software project, not a claim that models have already been trained.
> Dataset files, trained weights, CARLA, ROS 2, CUDA/TensorRT and recorded demo footage are
> intentionally not bundled. Download/install those separately as described below.

## Recommended dataset

Use **BDD100K** as the primary dataset. It provides varied driving conditions and annotations
for object detection, lane markings and drivable area. Add CARLA-generated frames for simulation.
Use CULane only if you want a dedicated lane-detection benchmark. KITTI is a good smaller
alternative for object detection/depth experiments, but BDD100K better matches this full project.

Official resources:
- BDD100K: https://bdd-data.berkeley.edu/
- KITTI: https://www.cvlibs.net/datasets/kitti/
- CULane: https://xingangpan.github.io/projects/CULane.html
- CARLA: https://carla.org/
- Ultralytics: https://docs.ultralytics.com/
- MiDaS: https://github.com/isl-org/MiDaS
- ROS 2: https://docs.ros.org/

## 1. Requirements

Recommended development environment:
- Windows 11 or Ubuntu 22.04/24.04
- Python 3.10 or 3.11
- 16 GB RAM recommended
- NVIDIA GPU recommended for training/inference, but CPU inference is supported
- For CARLA + ROS 2, Ubuntu is the smoother route. ROS 2 native Windows support is possible
  but setup is more involved.
- TensorRT requires a compatible NVIDIA GPU, CUDA stack and TensorRT installation.

## 2. Create environment (Windows CMD)

```bat
cd autonomous-driving-perception-stack
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python` is not recognized, install Python 3.10/3.11 and enable **Add Python to PATH**.
Check with `python --version` and `where python`.

## 3. Create environment (Ubuntu)

```bash
cd autonomous-driving-perception-stack
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Download and configure BDD100K

Download the BDD100K images and detection labels from the official site. Extract them outside
this repository or under `data/bdd100k/`. Do not commit the dataset to Git.

Expected layout (adapt paths if your downloaded archive differs):

```text
data/
  bdd100k/
    images/100k/
      train/
      val/
    labels/
      det_20/
        det_train.json
        det_val.json
```

Create a YOLO-format dataset using your own conversion script or a trusted BDD100K-to-YOLO
converter. The resulting layout should be:

```text
data/bdd100k_yolo/
  images/train/
  images/val/
  labels/train/
  labels/val/
  bdd100k.yaml
```

Example `bdd100k.yaml`:

```yaml
path: data/bdd100k_yolo
train: images/train
val: images/val
names:
  0: pedestrian
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motorcycle
  7: bicycle
  8: traffic light
  9: traffic sign
```

Important: verify the class IDs against the exact label conversion you use. The mapping above is
a common 10-class BDD100K detection mapping, but conversion tools can differ.

## 5. Inspect dataset balance

```bash
python scripts/dataset_stats.py --labels data/bdd100k_yolo/labels/train
```

This counts YOLO class instances and writes `outputs/class_balance.csv`.

## 6. Fine-tune YOLO

```bash
python scripts/train_detector.py --data data/bdd100k_yolo/data.yaml --model yolov8n.pt --epochs 3 --imgsz 416 --batch 4 --device 0 --workers 0 --fraction 0.2
```
For Validation
```bash
python scripts/validate_detector.py --weights "runs/detect/runs/detect/bdd100k_fast/weights/last.pt" --data data/bdd100k_yolo/data.yaml --batch 1 --max-det 100 --device 0
```

For a faster first test, use `--epochs 3`. The best checkpoint is normally saved by Ultralytics
under `runs/detect/train/weights/best.pt`. Copy it to `weights/best.pt` or pass its path directly
to the inference script.

## 7. Run the complete video perception pipeline

```bash
python scripts/run_pipeline.py --source "data/test_video.mp4" --weights "runs/detect/runs/detect/bdd100k_finetune/weights/last.pt" --output "outputs/perception_demo.mp4"
```

For a webcam:
```bash
python scripts/run_pipeline.py --source 0 --weights yolov8n.pt --output outputs/webcam_demo.mp4
```

The script overlays detection boxes, ByteTrack IDs, classical lane lines and approximate
MiDaS relative depth. MiDaS depth is **relative**, not metric distance in metres. Do not use it
as a safety-critical distance sensor.

## 8. Lane detection comparison

Classical OpenCV lane detection:
```bash
python scripts/lane_demo.py --source data/test_frame.jpg --output outputs/lane_classical.jpg
```

Optional segmentation model:
```bash
pip install transformers
python scripts/lane_segmentation.py --source data/test_frame.jpg --output outputs/lane_segmentation.jpg
```

The segmentation script uses a generic SegFormer checkpoint as an integration example. It is
not a lane-specific trained model. For a valid quantitative comparison, fine-tune a segmentation
model on BDD100K lane masks or CULane labels, then set its checkpoint and label mapping.
Do not interpret generic road-scene segmentation classes as lane markings without validation.

## 9. MiDaS depth

MiDaS weights are downloaded automatically by `torch.hub` on first use (internet required).
```bash
python scripts/depth_demo.py --source data/test_frame.jpg --output outputs/depth.jpg
```
Depth is ordinal/relative: closer regions generally receive different scores from farther regions.
It does not directly provide calibrated distance in metres.

## 10. CARLA simulation

Install CARLA separately using its official installation guide. Start the simulator, then run:

```bash
python scripts/carla_camera.py --host 127.0.0.1 --port 2000 --output outputs/carla_frames --frames 300
```

The script attaches an RGB camera to a spawned ego vehicle, saves frames and cleans up actors.
CARLA Python API version must match your CARLA release.

## 11. ROS 2 camera publisher (optional)

Install ROS 2 and source its environment. In a ROS 2 workspace, copy the `ros2/perception_camera`
package into `src/`, then build:

```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 run perception_camera camera_publisher
```

This publisher reads a local webcam by default and publishes `sensor_msgs/Image` on
`/camera/image_raw`. For CARLA ROS Bridge integration, install the official CARLA ROS Bridge
and subscribe to its camera topic instead of using this webcam publisher. ROS 2 is optional
for running the Python perception pipeline.

## 12. Export to ONNX and benchmark

```bash
python scripts/export_onnx.py --weights runs/detect/train/weights/best.pt --imgsz 640
python scripts/benchmark.py --weights runs/detect/train/weights/best.pt --source path/to/frame.jpg --runs 100
python scripts/benchmark.py --onnx runs/detect/train/weights/best.onnx --source path/to/frame.jpg --runs 100
```

Install ONNX Runtime:
```bash
pip install onnxruntime
```
For NVIDIA GPU ONNX Runtime:
```bash
pip install onnxruntime-gpu
```
TensorRT is platform/CUDA-specific and must be installed from NVIDIA's official instructions.
If TensorRT is available, Ultralytics export can create an engine:
```bash
yolo export model=runs/detect/train/weights/best.pt format=engine imgsz=640 half=True
```
Benchmark on the same machine, input size, precision and warm-up policy. Record GPU model,
driver, CUDA/TensorRT versions and batch size. A TensorRT engine is generally not portable
between arbitrary GPUs/software stacks.

## 13. Weights & Biases (optional)

```bash
pip install wandb
wandb login
python scripts/train_detector.py --data data/bdd100k_yolo/bdd100k.yaml --epochs 50 --wandb
```
Ultralytics may prompt for W&B integration or log through its callback integration. If that
integration is unavailable in your installed versions, training still runs; use the W&B API
in your own experiment wrapper to log metrics and artifacts.

## 14. Suggested execution order

1. Install Python dependencies and run `python scripts/self_check.py`.
2. Download BDD100K and convert labels to YOLO format.
3. Run dataset statistics and inspect class imbalance.
4. Fine-tune YOLO and save the best checkpoint.
5. Run video inference with ByteTrack.
6. Run classical lane detection; then train/evaluate a lane segmentation model.
7. Run MiDaS depth estimation and document its relative-depth limitation.
8. Install CARLA and capture synthetic camera frames.
9. Connect ROS 2 camera topics if required.
10. Export ONNX, benchmark PyTorch vs ONNX vs TensorRT where supported.
11. Record a CARLA demo and log experiment configuration/metrics in W&B.

## 15. Safety and research limitations

This repository is an educational/research prototype. It is not suitable for controlling a
real vehicle. Detection can fail, tracking IDs can switch, lanes can be missed, and monocular
depth is not metric without calibration. Keep all obstacle reactions inside CARLA or another
controlled simulator. Never connect this prototype to a real vehicle's steering, throttle or brake.

## Project structure

```text
autonomous-driving-perception-stack/
  configs/
  data/                 # downloaded data goes here 
  outputs/              # generated outputs
  scripts/
  src/perception/
  ros2/perception_camera/
  weights/              # trained checkpoints go here (not included)
  requirements.txt
  requirements-optional.txt
  README.md
```
