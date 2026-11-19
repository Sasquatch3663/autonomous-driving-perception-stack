
import cv2

video_path = "data/test_video.mp4"
output_path = "data/test_frame.jpg"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Cannot open driving video")

# Move to frame 100
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

ret, frame = cap.read()

if ret:
    cv2.imwrite(output_path, frame)
    print("Frame saved:", output_path)
else:
    print("Could not extract frame")

cap.release()
