import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scenarios import u_turn
from tracking import track

# U 弯，v = 15 m/s：三种方法同时出发，看谁贴着路线走
v = 15
path = u_turn()
methods = [("Pure Pursuit", "pp"), ("LQR", "lqr"), ("MPC N=10", "mpc")]
logs = [track(path, method, v, N=10) for name, method in methods]
n = max(len(log["x"]) for log in logs)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(path["x"], path["y"], "k--", linewidth=1, label="path")
lines = []
dots = []
widths = [1.5, 3.5, 1.5]
for (name, method), width in zip(methods, widths):
    # LQR 画粗一点，不然会被几乎重合的 MPC 线盖住
    line, = ax.plot([], [], linewidth=width, label=name)
    dot, = ax.plot([], [], "o", color=line.get_color())
    lines.append(line)
    dots.append(dot)
ax.set_xlim(-2, 32)
ax.set_ylim(-3, 20)
ax.set_aspect("equal")
ax.legend(loc="upper left", fontsize=8)
title = ax.set_title("")


def update(k):
    for log, line, dot in zip(logs, lines, dots):
        j = min(k, len(log["x"]) - 1)
        line.set_data(log["x"][:j + 1], log["y"][:j + 1])
        dot.set_data([log["x"][j]], [log["y"][j]])
    title.set_text(f"U turn at v = {v} m/s, t = {k * 0.1:.1f} s")
    return lines + dots + [title]


anim = FuncAnimation(fig, update, frames=n + 5, interval=100)
anim.save("results/uturn.gif", writer=PillowWriter(fps=10), dpi=80)
