from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _setup_axes(plotter: Callable[[np.ndarray, np.ndarray], plt.Axes]) -> Callable[[pd.DataFrame], None]:
    def wrapper(x: np.ndarray, y: np.ndarray):
        axes = plotter(x, y)
        axes.grid(axis='both', which='both')
        axes.set_xlim(0)
        axes.set_ylim(0)

    return wrapper


def scale_to_impulse(i_data: np.ndarray, conversion_peak: float) -> np.ndarray:
    pc_conversion = 1013.5
    k = pc_conversion / conversion_peak
    return i_data * k


@_setup_axes
def plot_scatter(x: np.ndarray, y: np.ndarray) -> plt.Axes:
    data = pd.DataFrame()

    data['x'] = x
    data['y'] = y

    return data.plot.scatter(x='x', y='y')


def compute_fermi(pc: np.ndarray, n: np.ndarray):
    return np.sqrt(n) / pc


if __name__ == '__main__':
    specter = pd.read_csv('./data/specter.csv')

    zero_level = 2.6867
    specter['N'] = np.abs(specter['N'].values - zero_level)
    plot_scatter(specter['I'].values, specter['N'].values)
    plt.show()

    conversion_peak = float(input("Введите ток, соответствующий конверсионным электронам: "))
    specter['pc'] = scale_to_impulse(specter['I'].values, conversion_peak)

    plot_scatter(specter['pc'], specter['N'])
    plt.show()

    # Fermi
    specter['Fermi'] = compute_fermi(specter['pc'].values, specter['N'].values)
    specter['E'] = np.sqrt(specter['pc'].values ** 2 + 511 ** 2) - 511
    plot_scatter(specter['E'], specter['Fermi'])
    plt.show()