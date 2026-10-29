import numpy as np
import matplotlib.pyplot as plt
from model import error_model

v = 5
dt = 0.1
L = 2.5
n = 20

A, B = error_model(v, dt, L)
x = np.array([0.5, 0.1])
u = np.array([0.0])
es = [x[0]]

# ---- 你来写：循环 n 次 ----
# 每一圈：用 A、B、x、u 算出新的 x，再把新 x 的 e 存进 es
for i in range(n):
    x = A @ x + B @ u
    es.append(x[0])


# ---- 下面画图 ----
print("最后的 e:", es[-1])
plt.plot(es, marker=".")
plt.xlabel("step")
plt.ylabel("lateral error e (m)")
plt.title("Drift with delta = 0")
plt.savefig("results/drift.png")
plt.show()
