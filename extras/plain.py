import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()


class Constants:
    m = 9.1e-31
    h = 1.05e-34
    a_max = 1.0e-5
    v_0_max = 8.0e-19


def transmission(a: float, v_0: float, e: float) -> float:
    def k_1(e: float) -> float:
        return np.sqrt(2 * Constants.m * e) / Constants.h

    def k_2(e: float) -> float:
        return np.sqrt(2 * Constants.m * np.abs(v_0 - e)) / Constants.h

    if e < v_0:
        return 1.0 / (1.0 + (np.sinh(k_2(e) * a)
                             * (k_1(e) / k_2(e) + k_2(e) / k_1(e))) ** 2)
    else:
        return 1.0 / (1.0 + (np.sin(k_2(e) * a)
                             * (k_1(e) / k_2(e) - k_2(e) / k_1(e))) ** 2)


def plot_transmission_height_energy():
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    n = 40
    barrier_height = np.linspace(0.1 * Constants.v_0_max, Constants.v_0_max)
    energy = np.linspace(0.1 * Constants.v_0_max, Constants.v_0_max * 2, n)

    x, y = np.meshgrid(barrier_height, energy)
    z = np.array([transmission(a=Constants.a_max, v_0=v_0, e=e) for (v_0, e)
                  in zip(np.ravel(x), np.ravel(y))])

    # Приводим к удобным единицам
    x = x / 1.6e-19
    y = y / 1.6e-19

    ax.scatter(x, y, z)

    ax.set_title('Ширина барьера, 10мкм')
    ax.set_xlabel('Высота барьера, эВ')
    ax.set_ylabel('Энергия частицы, эВ')
    ax.set_zlabel('Вероятность прохождения')


plot_transmission_height_energy()


def plot_transmission_width_energy():
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    n = 40
    barrier_width = np.linspace(0.1 * Constants.a_max, Constants.a_max)
    energy = np.linspace(0.1 * Constants.v_0_max, Constants.v_0_max * 2, n)

    x, y = np.meshgrid(barrier_width, energy)
    z = np.array([transmission(a=a, v_0=Constants.v_0_max, e=e) for (a, e)
                  in zip(np.ravel(x), np.ravel(y))])

    # Приводим к удобным единицам
    x = x / 1e-6
    y = y / 1.6e-19

    ax.scatter(x, y, z)

    ax.set_title('Высота барьера - 5 эВ')
    ax.set_xlabel('Ширина барьера, мкм')
    ax.set_ylabel('Энергия частицы, эВ')
    ax.set_zlabel('Вероятность прохождения')


plot_transmission_width_energy()

plt.show()
