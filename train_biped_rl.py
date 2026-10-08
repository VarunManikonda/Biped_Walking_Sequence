import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# =========================================================================
# 1. FIXED REINFORCEMENT LEARNING GYM ENVIRONMENT (NO-CHEAT MATRIX)
# =========================================================================
class Biped5DoFEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(Biped5DoFEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_path("biped_5dof.xml")
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        obs_shape = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        self.prev_action = np.zeros(10)

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Posture baseline settlement
        self.data.qpos[7:12] = [0.0, -0.4, 0.8, -0.4, 0.0]
        self.data.qpos[12:17] = [0.0, -0.4, 0.8, -0.4, 0.0]
        
        mujoco.mj_step(self.model, self.data)
        self.prev_action = np.zeros(10)
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action * 30.0
        mujoco.mj_step(self.model, self.data)

        # Extract strict positional metrics
        pelvis_x_velocity = self.data.qvel[0] # Direct forward velocity component
        pelvis_height = self.data.qpos[2]     # True vertical height
        
        # Extract orientation rotation matrix to check if body is upright
        # Index 3:7 contains the quaternion orientation [w, x, y, z]
        body_quaternion = self.data.qpos[3:7]
        tilt_penalty = -5.0 * (1.0 - body_quaternion[0]) # Penalize heavily if rotation vector deviates from vertical 1.0

        # -----------------------------------------------------------------
        # 🏆 FIXED MULTI-OBJECTIVE REWARD POSTURE MATRIX
        # -----------------------------------------------------------------
        # FIX 1: Max function blocks reward for flying backward
        r_forward = 5.0 * max(0.0, pelvis_x_velocity)  
        
        # FIX 2: Target strict upright height window
        r_height = -6.0 * abs(pelvis_height - 0.48)  
        
        # Energy and jitter dampening components
        r_torque = -0.01 * np.sum(np.square(action))  
        r_smooth = -0.05 * np.sum(np.square(action - self.prev_action))  

        # Combine variables
        reward = r_forward + r_height + r_torque + r_smooth + tilt_penalty
        self.prev_action = action.copy()

        # -----------------------------------------------------------------
        # 🛑 RIGID EVENT TERMINATION CONDITIONS
        # -----------------------------------------------------------------
        terminated = False
        
        # Fall event: Robot collapsed or inverted completely upside down
        if pelvis_height < 0.35 or body_quaternion[0] < 0.70:
            terminated = True
            reward -= 20.0 # Heavy deduction for crashing

        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}

if __name__ == "__main__":
    print("🧠 Initializing Un-cheatable PPO Agent Engine Core...")
    env = Biped5DoFEnv()
    
    # Using optimized exploration hyper-parameters
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=4096, batch_size=128, device="cpu", ent_coef=0.01)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=150000,
        save_path="./rl_checkpoints/",
        name_prefix="biped_fixed"
    )

    print("🏋️ Training Fixed Locomotion Model across 500,000 steps...")
    model.learn(total_timesteps=500000, callback=checkpoint_callback)
    
    model.save("ppo_biped_5dof_master")
    print("✅ Fixed weight file 'ppo_biped_5dof_master.zip' generated.")
