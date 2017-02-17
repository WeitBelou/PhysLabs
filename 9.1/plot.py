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


@_setup_axes
def plot_freq(T: np.ndarray, f_0: np.ndarray, f: np.ndarray):
    data = pd.DataFrame()
    T_label = '$T$, K'
    f_0_label = '$f_0$, Гц'
    f_label = '$f$, Гц'

    data[T_label] = T
    data[f_0_label] = f_0
    data[f_label] = f

    ax = data.plot.scatter(x=T_label, y=f_0_label, c='red')
    data.plot.scatter(ax=ax, x=T_label, y=f_label, c='blue')

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    ax.set_title('Зависимость $f, f_0$ от $T$.')

    return ax


@_setup_axes
def plot_kappa(T: np.array, f_0: np.array, f: np.array):
    data = pd.DataFrame()
    T_label = '$T$, K'
    kappa_label = '$\\kappa$'

    data[T_label] = T
    data[kappa_label] = f ** 2 / (f_0 ** 2 - f ** 2)

    ax = data.plot.scatter(x=T_label, y=kappa_label, c='red')

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    ax.set_title('Зависимость $\\frac{f^2}{f_0^2 - f^2}$ от $T$.')

    return ax


def _compute_temperature(V: np.array):
    t_0 = 21 + 273
    return V / 41 + t_0


def main():
    resonance_freq = pd.read_csv('./data/res_freq.csv')

    T = _compute_temperature(resonance_freq['V'])
    f_0 = resonance_freq['f_0']
    f = resonance_freq['f']

    plot_freq(T, f_0, f)
    plot_kappa(T, f_0, f)


if __name__ == '__main__':
    main()
