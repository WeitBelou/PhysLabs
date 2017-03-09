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
    f_0_label = '$f_0$, кГц'
    f_label = '$f$, кГц'

    data[T_label] = T
    data[f_0_label] = f_0
    data[f_label] = f

    ax = data.plot.scatter(x=T_label, y=f_0_label, c='red')
    data.plot.scatter(ax=ax, x=T_label, y=f_label, c='blue')

    ax.set_title('Зависимость $f, f_0$ от $T$.')

    return ax


@_setup_axes
def plot_inv_kappa(T: np.array, f_0: np.array, f: np.array):
    def compute_intersection(x: np.ndarray, y: np.ndarray) -> np.poly1d:
        fit = np.polyfit(x[4:-4], y[4:-4], 1)
        return np.poly1d(fit)

    data = pd.DataFrame()
    T_label = '$T$, K'
    kappa_label = '$\\frac{1}{\\kappa}$'

    data[T_label] = T
    data[kappa_label] = f ** 2 / (f_0 ** 2 - f ** 2)

    ax = data.plot.scatter(x=T_label, y=kappa_label, c='red')

    fitted_data = pd.DataFrame()
    linear_fit = compute_intersection(T, data[kappa_label])
    zero = -linear_fit[0] / linear_fit[1]
    print(zero)
    fitted_data[T_label] = np.linspace(zero, max(T), 100)
    fitted_data[kappa_label] = linear_fit(fitted_data[T_label])

    fitted_data.plot.line(x=T_label, y=kappa_label, c='blue', ax=ax)

    ax.set_title('Зависимость $\\frac{f^2}{f_0^2 - f^2}$ от $T$.')
    ax.set_xlim(min(T))

    return ax

@_setup_axes
def plot_kappa(T: np.array, f_0: np.array, f: np.array):
    data = pd.DataFrame()
    T_label = '$T$, K'
    kappa_label = '$\\frac{1}{\\kappa}$'

    data[T_label] = T
    data[kappa_label] = (f_0 ** 2 - f ** 2) / f ** 2

    ax = data.plot.scatter(x=T_label, y=kappa_label, c='red')

    ax.set_title('Зависимость $\\frac{f_0^2 - f^2}{f^2}$ от $T$.')

    return ax

def _compute_temperature(V: np.array):
    t_0 = 21 + 273
    delta_V = 90
    return (V - delta_V) / 41 + t_0


def main():
    resonance_freq = pd.read_csv('./data/res_freq.csv')

    T = _compute_temperature(resonance_freq['V'])
    f_0 = resonance_freq['f_0']
    f = resonance_freq['f']

    plot_freq(T, f_0, f)
    plot_inv_kappa(T, f_0, f)
    plot_kappa(T, f_0, f)


if __name__ == '__main__':
    main()
