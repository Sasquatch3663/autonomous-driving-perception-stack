import argparse, time, statistics, cv2
from ultralytics import YOLO

p=argparse.ArgumentParser()
p.add_argument("--weights"); p.add_argument("--onnx"); p.add_argument("--source",required=True)
p.add_argument("--runs",type=int,default=100); p.add_argument("--imgsz",type=int,default=640)
a=p.parse_args()
model_path=a.onnx or a.weights
if not model_path: raise SystemExit("Provide --weights or --onnx")
im=cv2.imread(a.source)
if im is None: raise SystemExit("Could not read source image")
model=YOLO(model_path)
for _ in range(10): model.predict(im,imgsz=a.imgsz,verbose=False)
times=[]
for _ in range(a.runs):
    t=time.perf_counter(); model.predict(im,imgsz=a.imgsz,verbose=False)
    times.append((time.perf_counter()-t)*1000)
avg=statistics.mean(times)
print(f"Model: {model_path}")
print(f"Runs: {a.runs} | mean latency: {avg:.2f} ms | throughput: {1000/avg:.2f} FPS")
print("Note: end-to-end FPS depends on decode, preprocessing, tracking, depth and hardware.")
