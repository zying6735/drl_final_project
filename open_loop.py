import numpy as np
import matplotlib.pyplot as plt
from pendulum_robot_env import PendulumRobotEnv

# === Create environment ===
env = PendulumRobotEnv()
env.reset()

# === Open-loop constant torque ===
tau_value = 0.5  # normalized action in [-1, 1]
n_steps = 3000
phi_values = []

for i in range(n_steps):
    if i>100:
        tau_value = 0
    state, _, _, _ = env.step([tau_value])
    phi_values.append(state[0])  # record phi

# Unwrap phi values
phi_values_unwrapped = np.unwrap(phi_values)

# === Plot phi over time ===
plt.figure(figsize=(10, 4))
plt.plot(phi_values_unwrapped, label='phi (unwrapped)')
plt.xlabel('Time step')
plt.ylabel('Phi (rad)')
plt.title('Open-loop response of phi under constant torque')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig("phi_plot.png")  # or .pdf, .svg, etc.
print("Plot saved as phi_plot.png")