import numpy as np
import dynamics_GP


class LEARNER():
    def __init__(self, env):
        self.firstIter = 1
        self.count = 1
        self.env = env
        self.torque_bound = 15.
        self.max_speed = 60.

        # Set up observation space and action space
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        print('Observation space', self.observation_space)
        print('Action space', self.action_space)

        # Determine dimensions of observation & action space
        self.observation_size = self.env.observation_space.shape[0]-2
        self.action_size = self.action_space.shape[0]


        # Build GP model of dynamics
        dynamics_GP.build_GP_model(self)