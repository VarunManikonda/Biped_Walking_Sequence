import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# =========================================================================
# 1. ENHANCED REINFORCEMENT LEARNING GYM ENVIRONMENT WRAPPER
# =========================================================================
class MasterBipedEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(MasterBipedEnv, self).__init__()
        # Load our verified 5+5 DoF MJCF sandbox asset model
        self.model = mujoco.MjModel.from_xml_path("biped_5dof.xml")
        self.data = mujoco.MjData(self.model)

        # 10 output motors bounded tightly between -1.0 and 1.0 torque vectors
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        
        # State observation dimensions (nq positions + nv velocities)
        obs_shape = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        
        self.prev_action = np.zeros(10)
        self.step_counter = 0

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Enforce stable, bended initial standing posture parameters
        self.data.qpos[7:12] = [0.0, -0.4, 0.8, -0.4, 0.0]   # Left Leg
        self.data.qpos[12:17] = [0.0, -0.4, 0.8, -0.4, 0.0]  # Right Leg
        
        mujoco.mj_step(self.model, self.data)
        self.prev_action = np.zeros(10)
        self.step_counter = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_counter += 1
        
        # Clamp action outputs to prevent numerical explosion bugs
        clipped_action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = clipped_action * 25.0  # Safe max torque ceiling
        
        # Step world physics forward by native 0.002s increments
        mujoco.mj_step(self.model, self.data)

        # FIXED: Extract explicit scalar values from vectors using clean indexing
        pelvis_forward_vel = float(self.data.qvel[0])  # Linear X Velocity
        pelvis_height = float(self.data.qpos[2])       # Vertical Z Height
        body_quat_w = float(self.data.qpos[3])         # Real component of the base quaternion

        # -----------------------------------------------------------------
        # 🏆 MATHEMATICAL SCORECARD: HIGH-FIDELITY LOCOMOTION METRICS
        # -----------------------------------------------------------------
        # Reward ONLY true forward displacement progression along the floor grid tile X-axis
        r_forward = 8.0 * max(0.0, pelvis_forward_vel)
        
        # Keep the pelvis hovering closely to an ideal 46cm standing plane clearance window
        r_height = -6.0 * abs(pelvis_height - 0.46)
        
        # Severely penalize tilting, cartwheeling, or rolling out of the vertical frame
        r_upright = -15.0 * (1.0 - body_quat_w)
        
        # Penalize jerky, spastic joint firing to encourage clean organic trajectories
        r_torque = -0.02 * np.sum(np.square(clipped_action))
        r_smooth = -0.08 * np.sum(np.square(clipped_action - self.prev_action))

        # Sum the multi-objective fitness scores
        reward = r_forward + r_height + r_upright + r_torque + r_smooth
        self.prev_action = clipped_action.copy()

        # -----------------------------------------------------------------
        # 🛑 UNCOMPROMISING TERMINATION FILTERS
        # -----------------------------------------------------------------
        terminated = False
        
        # Halt instantly if the biped collapses or the trunk tilts past 25 degrees
        if pelvis_height < 0.34 or body_quat_w < 0.90:
            terminated = True
            reward -= 50.0  # Deep penalty deduction for failing to stand
            
        # Stop numerical explosion warnings early
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isinf(self.data.qpos)):
            terminated = True
            reward -= 100.0

        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}

# =========================================================================
# 2. INTITALIZE PPO TRAINING ENGINE WITH PROGRESS ARCHIVING CALLBACKS
# =========================================================================
if __name__ == "__main__":
    print("🧠 Initializing Master Neural Network Policy (Isolated Core)...")
    env = MasterBipedEnv()
    
    # Custom tuned PPO hyperparameters optimized for high speed CPU vector processing
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.00025, 
        n_steps=4096, 
        batch_size=128, 
        device="cpu", 
        ent_coef=0.01,
        gae_lambda=0.95
    )
    
    # Establish automated background progress checkpoint writer
    checkpoint_callback = CheckpointCallback(
        save_freq=250000,
        save_path="./rl_master_checkpoints/",
        name_prefix="biped_master"
    )

    print("\n🏋️ MASTER RUN ACTIVATED: 3,000,000 TIMESTEPS OVERNIGHT BATCH RUN.")
    print("🤖 Go get some sleep! Progress files are automatically saving into './rl_master_checkpoints/'.\n")
    
    # Engage optimization processing pipeline
    model.learn(total_timesteps=3000000, callback=checkpoint_callback)
    
    # Save the absolute final weight package
    model.save("ppo_biped_master_final")
    print("\n🎉 MASTER TRAINING COMPLETE! 'ppo_biped_master_final.zip' successfully compiled.")
