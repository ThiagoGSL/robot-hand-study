import time
import mujoco
import mujoco.viewer
import numpy as np

def main():
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    
    print("Launching MuJoCo Viewer!")
    print("-> Use the 'Control' panel on the right to slide the motor values.")
    print("-> Press 'Space' to pause/play the physics.")
    
    # Launch the interactive viewer. This blocks until the user closes the window.
    mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    main()
