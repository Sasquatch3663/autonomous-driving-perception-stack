import importlib.util
mods = ["cv2", "numpy", "torch", "ultralytics", "yaml"]
failed = []
for name in mods:
    ok = importlib.util.find_spec(name) is not None
    print(f"{name}: {'OK' if ok else 'MISSING'}")
    if not ok: failed.append(name)
if failed:
    raise SystemExit("Install missing packages: " + ", ".join(failed))
print("Basic environment check passed. This does not verify CUDA, CARLA or ROS 2.")
