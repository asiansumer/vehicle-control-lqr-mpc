import numpy as np


def step(x, y, theta, v, delta, dt=0.1, L=2.5):
    # 运动学自行车模型走一步（和 P3 的 bicycle.step 一样）
    new_x = x + v * dt * np.cos(theta)
    new_y = y + v * dt * np.sin(theta)
    new_theta = theta + v * dt / L * np.tan(delta)
    return new_x, new_y, new_theta
