import time
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from train_biped_clean import CleanBipedEnv

print("🧠 Loading the fresh, un-corrupted AI brain file...")
env = CleanBipedEnv()
model = PPO.load("ppo_biped_clean_brain.zip", device="cpu")

print("🌐 Launching 3D Viewport window...")
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    viewer.cam.distance = 2.0
    viewer.cam.lookat = [0, 0, 0.4]
    
    while viewer.is_running():
        obs = env._get_obs()
        action, _states = model.predict(obs, deterministic=True)
        env.data.ctrl[:] = action * 25.0
        mujoco.mj_step(env.model, env.data)
        
        if env.data.qpos[2] < 0.35 or env.data.qpos[3] < 0.85:
            env.reset()
            
        viewer.sync()
        time.sleep(env.model.opt.timestep)
