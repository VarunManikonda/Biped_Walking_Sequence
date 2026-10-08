import os
import time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from train_biped_rl import Biped5DoFEnv

print("🧠 Loading the trained Reinforcement Learning AI model weights on CPU...")
model_path = "ppo_biped_5dof.zip"
if not os.path.exists(model_path):
    model_path = "ppo_biped_5dof"

# Initialize environment and force the weights to load onto the CPU
env = Biped5DoFEnv()
model = PPO.load(model_path, device="cpu")

print("🌐 Launching 3D Viewport window...")
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    viewer.cam.distance = 2.0
    viewer.cam.lookat = [0, 0, 0.4]
    viewer.cam.elevation = -15
    viewer.cam.azimuth = 135
    
    while viewer.is_running():
        obs = env._get_obs()
        action, _states = model.predict(obs, deterministic=True)
        env.data.ctrl[:] = action * 30.0
        mujoco.mj_step(env.model, env.data)
        
        pelvis_height = env.data.qpos[2]
        if pelvis_height < 0.35:
            env.reset()
            
        viewer.sync()
        time.sleep(env.model.opt.timestep)
