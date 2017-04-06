import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# All values in SI
PLANK_CONSTANT = 6.6e-4

# Some maximal parameters
V_0_MAX = 5
A_MAX = 10


def transmission(v_0: float, energy: float, width: float) -> float:
    m = 9.1e-31
    if energy <= 0:
        return 0.0
    elif energy < v_0:
        k = np.sqrt(2 * m * (v_0 - energy) / PLANK_CONSTANT ** 2)
        return (v_0 * np.sinh(k * width)) ** 2 / (4 * energy * (v_0 - energy)) + 1
    elif energy > v_0:
        k = np.sqrt(2 * m * (energy - v_0) / PLANK_CONSTANT ** 2)
        return (v_0 * np.sin(k * width)) ** 2 / (4 * energy * (energy - v_0)) + 1
    else:
        return 1 / (1 + m * width ** 2 * v_0 / (2 * PLANK_CONSTANT ** 2))


def main():
    n_points = 100

    energy = np.linspace(0.0, V_0_MAX * 2, n_points)
    v_0 = np.linspace(0.0, V_0_MAX, n_points)
    width = np.linspace(0.0, A_MAX, n_points)

    X, Y = np.meshgrid(energy, v_0)
    Z = np.array([transmission(v, e, A_MAX) for (v, e) in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)

    ax.plot_surface(X=X, Y=Y, Z=Z, cmap=cm.coolwarm)

    ax.set_xlabel('particle_energy, m')
    ax.set_ylabel('barrier height, J')
    ax.set_zlabel('transmission probablility')

    ax.set_xlim()
    ax.set_ylim()

    plt.show()


if __name__ == '__main__':
    main()
