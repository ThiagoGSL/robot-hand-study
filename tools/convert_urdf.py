import os
import mujoco

def main():
    urdf_path = "dexhand_description/urdf/dexhand-right.urdf"
    mod_urdf_path = "dexhand-right-mujoco.urdf"
    xml_path = "dexhand-right.xml"
    
    # 1. Read URDF and modify mesh paths
    with open(urdf_path, "r") as f:
        urdf_content = f.read()
        
    # Replace package:// with relative path to current dir
    urdf_content = urdf_content.replace("package://dexhand_description/", "./dexhand_description/")
    
    # Fix zero inertias which MuJoCo rejects
    urdf_content = urdf_content.replace('ixx="0.0"', 'ixx="1e-06"')
    urdf_content = urdf_content.replace('iyy="0.0"', 'iyy="1e-06"')
    urdf_content = urdf_content.replace('izz="0.0"', 'izz="1e-06"')
    
    # Write the modified URDF
    with open(mod_urdf_path, "w") as f:
        f.write(urdf_content)
        
    print(f"Saved modified URDF to {mod_urdf_path}")
    
    # 2. Load with MuJoCo
    model = mujoco.MjModel.from_xml_path(mod_urdf_path)
    
    # 3. Save as MJCF
    mujoco.mj_saveLastXML(xml_path, model)
    print(f"Successfully saved MJCF to {xml_path}")

if __name__ == "__main__":
    main()
