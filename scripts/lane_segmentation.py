"""Optional generic SegFormer visualization; not a lane-specific trained model."""
import argparse, cv2, numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch

p=argparse.ArgumentParser()
p.add_argument("--source",required=True); p.add_argument("--output",default="outputs/lane_segmentation.jpg")
p.add_argument("--model",default="nvidia/segformer-b0-finetuned-cityscapes-512-1024")
a=p.parse_args()
img=cv2.imread(a.source)
if img is None: raise SystemExit("Could not read image")
processor=SegformerImageProcessor.from_pretrained(a.model)
model=SegformerForSemanticSegmentation.from_pretrained(a.model).eval()
inputs=processor(images=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),return_tensors="pt")
with torch.inference_mode(): logits=model(**inputs).logits
pred=torch.nn.functional.interpolate(logits,size=img.shape[:2],mode="bilinear",align_corners=False).argmax(1)[0].numpy()
# Visualize semantic class IDs. This is NOT a lane mask.
mask=np.zeros_like(img); mask[:,:,1]=(pred.astype("uint8")*13)%255
out=cv2.addWeighted(img,0.65,mask,0.35,0)
cv2.imwrite(a.output,out)
print("Saved generic semantic segmentation visualization. Fine-tune a lane-specific model for lane comparison.")
