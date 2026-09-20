import time
import mujoco
import numpy as np
import cv2

def main():
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    
    # Disable gravity for 1 second
    model.opt.gravity[:] = 0.0
    
    # Create renderer
    renderer = mujoco.Renderer(model, 480, 640)
    
    # Step for 1 second to close fingers
    for step in range(200):
        for i in range(model.nu):
            data.ctrl[i] = 100.0
        mujoco.mj_step(model, data)
        
    # Render and save
    renderer.update_scene(data)
    img = renderer.render()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test_1s.png", img_bgr)
    
    print("Saved test_1s.png")

if __name__ == "__main__":
    main()
