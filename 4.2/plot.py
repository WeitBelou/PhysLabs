from typing import Callable, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rc

rc('font', **{'family': 'serif'})
rc('text', usetex=True)
rc('text.latex', unicode=True)
rc('text.latex', preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex', preamble=r'\usepackage[russian]{babel}')


def _setup_axes(plotter: Callable[[Any], plt.Axes]) -> Callable[[Any], None]:
    def wrapper(*args, **kwargs):
        ax = plotter(*args, **kwargs)
        ax.grid(axis='both', which='both')
        plt.show()

    return wrapper


def _scale_to_impulse(i_data: np.ndarray, conversion_peak: float) -> np.ndarray:
    pc_conversion = 1013.5
    k = pc_conversion / conversion_peak
    return i_data * k


@_setup_axes
def plot_fermi(pc: np.ndarray, n: np.ndarray):
    def compute_fermi(pc: np.ndarray, n: np.ndarray) -> np.ndarray:
        return np.sqrt(n) / pc

    def compute_energy(pc: np.ndarray) -> np.ndarray:
        return np.sqrt(pc.values ** 2 + 511 ** 2) - 511

    data = pd.DataFrame()
    y_label = '$\\frac{\\sqrt{N(p)}}{p}$, 1 / кэВ'
    x_label = '$E$, кэВ'

    data[x_label] = compute_energy(pc)
    data[y_label] = compute_fermi(pc, n)
    ax = data.plot.scatter(x=x_label, y=y_label)

    poly = np.polyfit(data[x_label].values[8:17],
                      data[y_label].values[8:17], 1)
    print(poly)
    print(- poly[1] / poly[0])
    x_data = np.linspace(150, 600)
    y_data = np.polyval(poly, x_data)
    plt.plot(x_data, y_data, axes=ax)

    ax.set_title('График Ферми-Кюри')

    ax.set_ylim(0, None)
    ax.set_xlim(0, None)

    ax.set_ylim(0, np.max(data[y_label]) * 1.1)

    return ax


@_setup_axes
def plot_specter(pc: np.ndarray, n: np.ndarray):
    data = pd.DataFrame()
    x_label = '$pc, кЭв$'
    y_label = '$N, с$'

    data[x_label] = pc
    data[y_label] = n

    ax = data.plot.scatter(x=x_label, y=y_label)

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    ax.set_title('\\beta-спектр')

    return ax


def main():
    specter = pd.read_csv('./data/specter.csv')

    # Фон
    zero_level = 2.6867
    specter['N'] = np.abs(specter['N'].values - zero_level)

    # Здесь устанавливается ток соответствующий конверсионному пику
    conversion_peak = 3.0515
    specter['pc'] = _scale_to_impulse(specter['I'].values, conversion_peak)

    # Fermi
    plot_fermi(specter['pc'], specter['N'])


if __name__ == '__main__':
    main()
