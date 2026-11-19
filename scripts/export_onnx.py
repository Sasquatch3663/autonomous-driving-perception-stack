import argparse
from ultralytics import YOLO
p=argparse.ArgumentParser()
p.add_argument("--weights",required=True); p.add_argument("--imgsz",type=int,default=640)
p.add_argument("--half",action="store_true")
a=p.parse_args()
model=YOLO(a.weights)
path=model.export(format="onnx",imgsz=a.imgsz,half=a.half,dynamic=False)
print("ONNX exported:",path)
