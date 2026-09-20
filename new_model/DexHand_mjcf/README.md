# DexHand_mjcf

MuJoCo model exported by sw2robot from the SolidWorks assembly `DexHand`.

    mjcf/DexHand.xml   the model
    mjcf/assets/      its binary-STL mesh assets

Load it with:

    import mujoco
    model = mujoco.MjModel.from_xml_path("mjcf/DexHand.xml")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)   # the "home" keyframe

What the exporter derived from the CAD, rather than guessed:

* joint damping from each joint's own `effort / velocity` (N*m*s/rad): 3.185 on 38 joints
* contact spheres sized from each foot's own contact patch: `c_6x10x3_10_toe` r=0.001511 m, `c_6x10x3_12_toe` r=0.001511 m, `c_6x10x3_16_toe` r=0.001518 m, `c_6x10x3_17_toe` r=0.00151 m, `c_6x10x3_18_toe` r=0.001659 m, `c_6x10x3_20_toe` r=0.001636 m, `c_6x10x3_22_toe` r=0.001659 m, `c_6x10x3_3_toe` r=0.001977 m, `c_6x10x3_4_toe` r=0.001977 m, `c_6x10x3_42_toe` r=0.001631 m, `c_6x10x3_44_toe` r=0.001679 m, `c_6x10x3_46_toe` r=0.001678 m, `c_6x10x3_48_toe` r=0.001653 m, `c_6x10x3_50_toe` r=0.00183 m, `c_6x10x3_52_toe` r=0.001831 m, `c_6x10x3_56_toe` r=0.00183 m, `c_6x10x3_57_toe` r=0.001712 m, `c_6x10x3_6_toe` r=0.001425 m, `c_6x10x3_8_toe` r=0.001979 m
* IMU site `imu_in_body` and sensors `imu_ang_vel`, `imu_lin_vel`, `imu_lin_acc`, `root_angmom`
* `home` keyframe with the base at -0.022 m, the height at which nothing is below the floor
