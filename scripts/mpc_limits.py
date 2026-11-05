import numpy as np
import matplotlib.pyplot as plt
from model import error_model
from lqr import lqr_gain
from mpc import mpc_inputs_limited

# 车从路线左边 4 米出发，方向盘最多打 35°
v = 5
dt = 0.1
L = 2.5
n = 60
N = 20
x0 = np.array([4.0, 0.0])
u_max = np.radians(35)

A, B = error_model(v, dt, L)
Q = np.eye(2)
R = np.array([[0.1]])
K = lqr_gain(A, B, Q, R)

fig, (ax_e, ax_d) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for name in ["LQR + clip", "MPC with limit"]:
    x = x0
    es = [x[0]]
    us = []
    cost = 0.0
    for k in range(n):
        if name == "LQR + clip":
            # LQR 先算，超出 35° 再硬截断
            u = np.clip(-(K @ x)[0], -u_max, u_max)
        else:
            # MPC 算的时候就知道最多只能打 35°
            u = mpc_inputs_limited(A, B, Q, R, N, x, u_max)[0, 0]
        x = A @ x + B @ np.array([u])
        es.append(x[0])
        us.append(u)
        cost = cost + x @ Q @ x + R[0, 0] * u * u
    es = np.array(es)
    settle = np.argmax(np.abs(es) < 0.05)
    print(f"{name:15s} 总罚款 {cost:7.2f}  冲过头 {-es.min():.3f} 米  e<0.05 用了 {settle} 步")
    ax_e.plot(es, marker=".", label=name)
    ax_d.plot(np.degrees(us), marker=".", label=name)

ax_e.axhline(0, color="gray", linewidth=0.8)
ax_e.set_ylabel("e (m)")
ax_e.legend()
ax_d.axhline(35, color="red", linestyle="--", linewidth=0.8)
ax_d.axhline(-35, color="red", linestyle="--", linewidth=0.8)
ax_d.set_ylabel("steering δ (deg)")
ax_d.set_xlabel("step")
ax_d.legend()
fig.tight_layout()
plt.savefig("results/mpc_limits.png", dpi=120)
plt.show()
