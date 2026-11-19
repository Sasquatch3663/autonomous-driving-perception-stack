import argparse, time
from pathlib import Path
import carla
import numpy as np
import cv2

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=2000)
    p.add_argument("--output",default="outputs/carla_frames"); p.add_argument("--frames",type=int,default=300)
    a=p.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    client=carla.Client(a.host,a.port); client.set_timeout(10.0)
    world=client.get_world(); bp=world.get_blueprint_library()
    vehicle_bp=bp.filter("vehicle.*")[0]
    spawn=world.get_map().get_spawn_points()
    if not spawn: raise SystemExit("CARLA map has no spawn points")
    vehicle=world.try_spawn_actor(vehicle_bp,spawn[0])
    if vehicle is None: raise SystemExit("Could not spawn vehicle; try another CARLA spawn point")
    camera=None; frames=[]
    try:
        cam_bp=bp.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x","1280"); cam_bp.set_attribute("image_size_y","720")
        cam_bp.set_attribute("fov","90")
        camera=world.spawn_actor(cam_bp,carla.Transform(carla.Location(x=1.5,z=2.4)),attach_to=vehicle)
        camera.listen(lambda image: frames.append(image))
        vehicle.set_autopilot(True)
        deadline=time.time()+120
        saved=0
        while saved<a.frames and time.time()<deadline:
            world.wait_for_tick()
            while frames and saved<a.frames:
                image=frames.pop(0)
                arr=np.frombuffer(image.raw_data,dtype=np.uint8).reshape((image.height,image.width,4))
                bgr=arr[:,:,:3][:,:,::-1].copy()
                cv2.imwrite(str(out/f"frame_{saved:06d}.jpg"),bgr); saved+=1
        print(f"Saved {saved} frames to {out}")
    finally:
        if camera:
            camera.stop(); camera.destroy()
        vehicle.destroy()
if __name__=="__main__": main()
