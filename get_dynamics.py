import numpy as np
import math

def get_dynamics(curr_state, action):
    dt = 0.05
    G = 10.
    m = 1.
    l = 1.
    obs = np.squeeze(curr_state)
    theta = np.arctan2(obs[1], obs[0])
    theta_dot = obs[2]
    f = np.array([-3 * G / (2 * l) * np.sin(theta + np.pi) * dt ** 2 + theta_dot * dt + theta + 3 / (
                m * l ** 2) * action * dt ** 2,
                  theta_dot - 3 * G / (2 * l) * np.sin(theta + np.pi) * dt + 3 / (m * l ** 2) * action * dt])
    g = np.array([3 / (m * l ** 2) * dt ** 2, 3 / (m * l ** 2) * dt])

    x = np.array([theta, theta_dot])
    return [np.squeeze(f), np.squeeze(g), np.squeeze(x)]
