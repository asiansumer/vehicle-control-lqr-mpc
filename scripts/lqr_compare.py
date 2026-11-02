import numpy as np
import matplotlib.pyplot as plt
from model import error_model
from control import closed_loop_matrix, simulate
from lqr import lqr_gain

v = 5
dt = 0.1
L = 2.5
n = 60
x0 = [0.5, 0.1]
max_steer = np.radians(35)

A, B = error_model(v, dt, L)
Q = np.eye(2)

# 第 2 关手调的 K，和三组不同 R 的 LQR
cases = [("hand K", np.array([[0.5, 2.0]]))]
for r in [10.0, 1.0, 0.1]:
    K = lqr_gain(A, B, Q, np.array([[r]]))
    cases.append((f"LQR R={r}", K))

fig, (ax_e, ax_d) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for name, K in cases:
    xs, us = simulate(A, B, K, x0, n)
    biggest = np.max(np.abs(np.linalg.eigvals(closed_loop_matrix(A, B, K))))
    settle = np.argmax(np.abs(xs[:, 0]) < 0.05)
    print(f"{name:12s} K={np.round(K[0], 3)}  max|λ|={biggest:.3f}  e<0.05 用了 {settle} 步  最大方向盘 {np.degrees(np.max(np.abs(us))):.1f}°")
    ax_e.plot(xs[:, 0], marker=".", label=name)
    ax_d.plot(np.degrees(us), marker=".", label=name)

ax_e.axhline(0, color="gray", linewidth=0.8)
ax_e.set_ylabel("e (m)")
ax_e.legend()
ax_d.axhline(np.degrees(max_steer), color="red", linestyle="--", linewidth=0.8)
ax_d.axhline(-np.degrees(max_steer), color="red", linestyle="--", linewidth=0.8)
ax_d.set_ylabel("steering δ (deg)")
ax_d.set_xlabel("step")
fig.tight_layout()
plt.savefig("results/lqr_compare.png", dpi=120)
plt.show()
