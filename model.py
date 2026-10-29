import numpy as np


def error_model(v, dt, L):
    A = np.array([ [1, dt*v], [0, 1]])
    B = np.array([ [0], [dt*v/L]])
    return (A, B)
