import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from stable_baselines3 import PPO

class CleanBipedEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(CleanBipedEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_path("biped_5dof.xml")
        self.data = mujoco.MjData(self.model)

        # 10 normalized actuator control limits
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        obs_shape = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        self.prev_action = np.zeros(10)

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Initial standard bent posture settlement
        self.data.qpos[7:12] = [0.0, -0.4, 0.8, -0.4, 0.0]
        self.data.qpos[12:17] = [0.0, -0.4, 0.8, -0.4, 0.0]
        
        mujoco.mj_step(self.model, self.data)
        self.prev_action = np.zeros(10)
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action * 25.0  # Reduced torque scale factor to stop wild shaking
        mujoco.mj_step(self.model, self.data)

        # Read specific state outputs directly from the arrays
        pelvis_forward_vel = self.data.qvel[0]  # Array index 0 tracks explicit X velocity
        pelvis_height = self.data.qpos[2]       # Array index 2 tracks explicit Z vertical height
        
        # Extract orientation values to prevent upside-down tumbling hacks
        body_quat_w = self.data.qpos[3]

        # -----------------------------------------------------------------
        # 🏆 EXPLICIT LOCOMOTION SCORECARD (NO REWARD HACKS ALLOWED)
        # -----------------------------------------------------------------
        # Reward the AI ONLY for moving forward along the positive X track
        r_forward = 6.0 * max(0.0, pelvis_forward_vel)
        
        # Penalize deviating from the 48cm clearance height window
        r_height = -5.0 * abs(pelvis_height - 0.48)
        
        # Penalize tilting or doing flips out of the vertical plane
        r_upright = -10.0 * (1.0 - body_quat_w)
        
        # Power smoothing terms to damp high frequency jitter shaking
        r_torque = -0.02 * np.sum(np.square(action))
        r_smooth = -0.06 * np.sum(np.square(action - self.prev_action))

        reward = r_forward + r_height + r_upright + r_torque + r_smooth
        self.prev_action = action.copy()

        # -----------------------------------------------------------------
        # 🛑 RIGID FALL AND FLY TERMINATION FILTERS
        # -----------------------------------------------------------------
        terminated = False
        
        # Trigger an instant crash reset if the pelvis falls or tilts past 30 degrees
        if pelvis_height < 0.35 or body_quat_w < 0.85:
            terminated = True
            reward -= 25.0  # Drop its score significantly for falling

        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}

if __name__ == "__main__":
    print("🧠 Spinning up a pristine, empty neural network (PPO Core)...")
    env = CleanBipedEnv()
    
    # Initialize a fresh model layout on CPU with zero prior memory files
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, device="cpu")
    
    print("🏋️ Training clean walking reflexes across 200,000 timesteps...")
    model.learn(total_timesteps=200000)
    
    model.save("ppo_biped_clean_brain")
    print("✅ Complete! Fresh weight file 'ppo_biped_clean_brain.zip' generated.")
