import time
import mujoco
import mujoco.viewer

print("🤖 Loading MuJoCo 3D Physics Viewport...")

# Load our verified 5+5 DoF configuration asset file
model = mujoco.MjModel.from_xml_path("biped_5dof.xml")
data = mujoco.MjData(model)

print("🌐 Launching native visual sandbox window...")
print("💡 TIP: Press 'Space' inside the graphics window to pause/unpause physics!")

# Open the interactive graphics window canvas
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set the initial camera parameters for a clean overview angle
    viewer.cam.distance = 2.0
    viewer.cam.lookat = [0, 0, 0.4]
    viewer.cam.elevation = -15
    viewer.cam.azimuth = 135
    
    start_time = time.time()
    
    while viewer.is_running():
        step_start = time.time()
        
        # -----------------------------------------------------------------
        # Real-time Dynamic Physics Step Execution
        # -----------------------------------------------------------------
        # MuJoCo steps the world forward by 0.002 seconds, computing true
        # gravitational pull, joint constraints, and floor impact collisions.
        mujoco.mj_step(model, data)
        
        # Synchronize and refresh the graphic viewer elements
        viewer.sync()
        
        # Maintain a steady simulation framerate matching the physics timestep
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
            
print("\n🛑 Viewport window closed by user.")
