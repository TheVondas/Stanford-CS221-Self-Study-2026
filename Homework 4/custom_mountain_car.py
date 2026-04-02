"""Custom Mountain Car environment that accepts max_speed as a constructor argument."""
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
import numpy as np
from gymnasium import spaces


class CustomMountainCarEnv(MountainCarEnv):
    def __init__(self, render_mode=None, goal_velocity=0, max_speed=0.07):
        super().__init__(render_mode=render_mode, goal_velocity=goal_velocity)
        self.max_speed = max_speed
        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
