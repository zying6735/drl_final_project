import numpy as np
import gym
from gym import spaces

class PendulumRobotEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # === Physical parameters (fill these with real values) ===
        self.m1 = 3.452
        self.m2 = 0.199
        self.R = 0.15
        self.l = 0.043
        self.J1 = 0.00002113461
        self.g = 9.81
        self.c = 0
        self.f = 0  # Can be a constant or function of state

        self.dt = 0.01
        self.phi_dot_target = 0.05

        # === Observation space: [phi, alpha, phi_dot, alpha_dot] ===
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.pi, -np.inf, -np.inf]),
            high=np.array([np.pi, np.pi, np.inf, np.inf]),
            dtype=np.float32
        )

        # === Action space: torque τ applied to shell ===
        self.torque_limit = 5  # Max absolute torque in Nm
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.zeros(4)
        return self.state

    def dynamics(self, state, tau):
        phi, alpha, phi_dot, alpha_dot = state

        M = self.m1 * self.R**2 + self.J1 + self.m2 * self.R**2
        C = self.m2 * self.R * self.l * np.cos(alpha)
        I = self.l**2

        rhs1 = tau - self.c * phi_dot - self.f + self.m2 * self.R * self.l * alpha_dot**2 * np.sin(alpha)
        rhs2 = -self.R * self.l * phi_dot * alpha_dot * (np.cos(alpha) + np.sin(alpha)) - self.g * self.l * np.sin(alpha)

        A = np.array([
            [M, C],
            [C, I]
        ])
        b = np.array([rhs1, rhs2])
        phi_ddot, alpha_ddot = np.linalg.solve(A, b)

        return np.array([phi_dot, alpha_dot, phi_ddot, alpha_ddot])

    def step(self, action):
        tau = float(np.clip(action[0], -1.0, 1.0)) * self.torque_limit

        # Runge-Kutta 4th Order Integration
        s = self.state
        dt = self.dt

        k1 = self.dynamics(s, tau)
        k2 = self.dynamics(s + 0.5 * dt * k1, tau)
        k3 = self.dynamics(s + 0.5 * dt * k2, tau)
        k4 = self.dynamics(s + dt * k3, tau)

        s_next = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Wrap angles to [-π, π]
        s_next[0] = (s_next[0] + np.pi) % (2 * np.pi) - np.pi
        s_next[1] = (s_next[1] + np.pi) % (2 * np.pi) - np.pi

        self.state = s_next

        reward = -abs(self.state[2] - self.phi_dot_target)  # phi_dot tracking
        done = False
        return self.state, reward, done, {"tau": tau}

    def render(self, mode='human'):
        print(f"phi={self.state[0]:.2f}, alpha={self.state[1]:.2f}, phi_dot={self.state[2]:.2f}, alpha_dot={self.state[3]:.2f}")

    def set_target_velocity(self, phi_dot_target):
        self.phi_dot_target = phi_dot_target