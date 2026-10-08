import time, meshcat, numpy as np, pinocchio as pin, pinocchio.visualize, pink
from pink.tasks import FrameTask

# 1. GENERATE THE SKELETON DEFINITION
urdf_string = """<robot name="biped_5dof">
  <link name="pelvis"><inertial><mass value="5"/><origin xyz="0 0 0"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial><visual><geometry><box size="0.12 0.25 0.08"/></geometry><material name="b"><color rgba="0.2 0.2 0.8 1"/></material></visual></link>
  <joint name="left_hip_roll" type="revolute"><parent link="pelvis"/><child link="left_thigh"/><origin xyz="0 0.1 0"/><axis xyz="1 0 0"/><limit effort="30" lower="-1.5" upper="1.5" velocity="5"/></joint>
  <link name="left_thigh"><inertial><mass value="1"/><origin xyz="0 0 -0.1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/></inertial><visual><origin xyz="0 0 -0.1"/><geometry><cylinder radius="0.02" length="0.2"/></geometry><material name="r"><color rgba="0.8 0.2 0.2 1"/></material></visual></link>
  <joint name="left_hip_pitch" type="revolute"><parent link="left_thigh"/><child link="left_shin"/><origin xyz="0 0 -0.2"/><axis xyz="0 1 0"/><limit effort="30" lower="-1.5" upper="1.5" velocity="5"/></joint>
  <link name="left_shin"><inertial><mass value="1"/><origin xyz="0 0 -0.1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/></inertial><visual><origin xyz="0 0 -0.1"/><geometry><cylinder radius="0.015" length="0.2"/></geometry><material name="r"/></visual></link>
  <joint name="left_knee_pitch" type="revolute"><parent link="left_shin"/><child link="left_tarsus"/><origin xyz="0 0 -0.2"/><axis xyz="0 1 0"/><limit effort="30" lower="-1.5" upper="1.5" velocity="5"/></joint>
  <link name="left_tarsus"><inertial><mass value="0.5"/><origin xyz="0 0 -0.1"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.001"/></inertial><visual><origin xyz="0 0 -0.1"/><geometry><cylinder radius="0.01" length="0.2"/></geometry><material name="r"/></visual></link>
  <joint name="left_ankle_pitch" type="revolute"><parent link="left_tarsus"/><child link="left_foot_roll_link"/><origin xyz="0 0 -0.2"/><axis xyz="0 1 0"/><limit effort="30" lower="-1.5" upper="1.5" velocity="5"/></joint>
  <link name="left_foot_roll_link"><inertial><mass value="0.2"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
  <joint name="left_ankle_roll" type="revolute"><parent link="left_foot_roll_link"/><child link="left_foot"/><axis xyz="1 0 0"/><limit effort="30" lower="-1.5" upper="1.5" velocity="5"/></joint>
  <link name="left_foot"><inertial><mass value="0.3"/><inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/></inertial><visual><origin xyz="0.02 0 -0.01"/><geometry><box size="0.12 0.06 0.02"/></geometry><material name="g"><color rgba="0.2 0.8 0.2 1"/></material></visual></link>
  <joint name="right_hip_roll" type="revolute"><parent link="pelvis"/><child link="right_thigh"/><origin xyz="0 -0.1 0"/><axis xyz="1 0 0"/><limit effort="30" lower="-1.5" upper="1.5" velocity="5"/></joint>
  <link name="right_thigh"><inertial><mass value="1"/><origin xyz="0 0 -0.1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/></inertial><visual><origin xyz="0 0 -0.1"/><geometry><cylinder radius="0.02" length="0.2"/></geometry><material name="r"/></visual></link>
  <joint name="right_hip_pitch" type="revolute"><parent link="right_thigh"/><child link="right_shin"/><origin xyz="0 0 -0.2"/><axis xyz="0 1 0"/><limit effort="30" lower="-1.5" upper="1.5" velocity="5"/></joint>
  <link name="right_shin"><inertial><mass value="1"/><origin xyz="0 0 -0.1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/></inertial><visual><origin xyz="0 0 -0.1"/><geometry><cylinder radius="0.015" length="0.2"/></geometry><material name="r"/></visual></link>
  <joint name="right_knee_pitch" type="revolute"><parent link="right_shin"/><child link="right_tarsus"/><origin xyz="0 0 -0.2"/><axis xyz="0 1 0"/><limit effort="30" lower="-1.5" upper="1.5" velocity="5"/></joint>
  <link name="right_tarsus"><inertial><mass value="0.5"/><origin xyz="0 0 -0.1"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.001"/></inertial><visual><origin xyz="0 0 -0.1"/><geometry><cylinder radius="0.01" length="0.2"/></geometry><material name="r"/></visual></link>
  <joint name="right_ankle_pitch" type="revolute"><parent link="right_tarsus"/><child link="right_foot_roll_link"/><origin xyz="0 0 -0.2"/><axis xyz="0 1 0"/><limit effort="30" lower="-1.5" upper="1.5" velocity="5"/></joint>
  <link name="right_foot_roll_link"><inertial><mass value="0.2"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
  <joint name="right_ankle_roll" type="revolute"><parent link="right_foot_roll_link"/><child link="right_foot"/><axis xyz="1 0 0"/><limit effort="30" lower="-1.5" upper="1.5" velocity="5"/></joint>
  <link name="right_foot"><inertial><mass value="0.3"/><inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0.0" izz="0.002"/></inertial><visual><origin xyz="0.02 0 -0.01"/><geometry><box size="0.12 0.06 0.02"/></geometry><material name="g"/></visual></link>
</robot>"""
with open("biped_5dof.urdf", "w") as f: f.write(urdf_string)

# 2. ENGAGE KINEMATICS OPTIMIZER ENGINE
model = pin.buildModelFromUrdf("biped_5dof.urdf", pin.JointModelFreeFlyer())
viz = pin.visualize.MeshcatVisualizer(model, pin.buildGeomFromUrdf(model, "biped_5dof.urdf", pin.GeometryType.VISUAL), pin.buildGeomFromUrdf(model, "biped_5dof.urdf", pin.GeometryType.VISUAL))
viz.initViewer(open=True); viz.loadViewerModel(); data = model.createData()
configuration = pink.Configuration(model, data, np.append(np.array([0, 0, 0.5, 0, 0, 0, 1]), pin.neutral(model)[7:]))

pelvis_task = FrameTask("pelvis", position_cost=1.0, orientation_cost=1.0)
left_foot_task = FrameTask("left_foot", position_cost=1.0, orientation_cost=0.5)
right_foot_task = FrameTask("right_foot", position_cost=1.0, orientation_cost=0.5)

pelvis_init = configuration.get_transform_frame_to_world("pelvis")
left_init = configuration.get_transform_frame_to_world("left_foot")
right_init = configuration.get_transform_frame_to_world("right_foot")
pelvis_task.set_target(pelvis_init); viz.display(configuration.q)

left_foot_x, right_foot_x, pelvis_x, dt, t = float(left_init.translation[0]), float(right_init.translation[0]), float(pelvis_init.translation[0]), 0.01, 0.0

# 3. CONTROL RUN LOOP (WITH SIMULATED TILT DISTURBANCE)
print("🚀 Running active feedback loops smoothly! Check browser window.")
try:
    while True:
        phase = np.sin(2 * np.pi * 0.2 * t)
        
        simulated_wind_push = 0.08 if 3.0 <= t <= 5.0 else 0.0
        current_body_roll = float(configuration.q[3]) + simulated_wind_push
        
        Kp = 0.45  # Feedback correction multiplier
        sensor_correction = -Kp * current_body_roll
        
        y_shift = (0.04 * phase) + sensor_correction
        pelvis_x += 0.0003
        
        pelvis_task.set_target(pin.SE3(pelvis_init.rotation, np.array([pelvis_x, y_shift, float(pelvis_init.translation[2])])))
        
        z_l = float(left_init.translation[2]) + 0.04 * ((-phase - 0.3) / 0.7) if phase < -0.3 else float(left_init.translation[2])
        if phase < -0.3: left_foot_x += 0.0012
            
        z_r = float(right_init.translation[2]) + 0.04 * ((phase - 0.3) / 0.7) if phase > 0.3 else float(right_init.translation[2])
        if phase > 0.3: right_foot_x += 0.0012
            
        left_foot_task.set_target(pin.SE3(left_init.rotation, np.array([left_foot_x, float(left_init.translation[1]), z_l])))
        right_foot_task.set_target(pin.SE3(right_init.rotation, np.array([right_foot_x, float(right_init.translation[1]), z_r])))
        
        velocity = pink.solve_ik(configuration, [pelvis_task, left_foot_task, right_foot_task], dt, solver="proxqp", safety_break=False)
        configuration.integrate_inplace(velocity, dt); viz.display(configuration.q)
        
        if 3.0 <= t <= 5.0:
            print(f"⚠️ [IMU ALERT] Roll Tilt: {current_body_roll:.3f} rad | Kp Adjusting Balance By: {sensor_correction:.3f}m")
        else:
            print(f"Time: {t:.1f}s | Status: Stably Balanced")
            
        time.sleep(dt); t += dt
except KeyboardInterrupt:
    print("\n🛑 Stopped.")
