import numpy as np
import matplotlib.pyplot as plt
from model import error_model
from control import closed_loop_matrix, simulate

v = 5
dt = 0.1
L = 2.5
n = 60
x0 = [0.5, 0.1]

A, B = error_model(v, dt, L)

# 四组 K = [k_e, k_θ]，前一个管“离线多远”，后一个管“车头歪多少”
cases = [
    ("only heading", [0.0, 1.0]),
    ("only offset", [0.5, 0.0]),
    ("both", [0.5, 2.0]),
    ("too aggressive", [2.0, 12.0]),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, (name, k) in zip(axes.ravel(), cases):
    K = np.array([k])
    xs, us = simulate(A, B, K, x0, n)
    eig = np.linalg.eigvals(closed_loop_matrix(A, B, K))
    biggest = np.max(np.abs(eig))
    print(f"{name:15s} K={k}  |λ|最大={biggest:.3f}  第20步 e={xs[20, 0]:.3f}  第60步 e={xs[-1, 0]:.3g}")
    ax.plot(xs[:, 0], marker=".")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(f"{name}: K={k}, max|λ|={biggest:.3f}")
    ax.set_xlabel("step")
    ax.set_ylabel("e (m)")
    if biggest > 1:
        ax.set_ylim(-3, 3)

fig.tight_layout()
plt.savefig("results/feedback.png", dpi=120)
plt.show()
