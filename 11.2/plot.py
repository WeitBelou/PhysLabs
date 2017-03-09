from typing import Callable, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _setup_axes(plotter: Callable[[Any], plt.Axes]) -> Callable[[Any], None]:
    def wrapper(*args, **kwargs):
        ax = plotter(*args, **kwargs)
        ax.grid(axis='both', which='both')
        plt.show()

    return wrapper


@_setup_axes
def plot_specter(v_from_alpha: np.ndarray, v_0: float, r: float):
    def _convert_from_alpha_to_wavelength(alpha: np.ndarray):
        return alpha

    def _convert_to_i(v: np.ndarray, r: float) -> np.ndarray:
        return v / r

    def _normalize_to_bulb_specter(i: np.ndarray) -> np.ndarray:
        bulb_specter = pd.read_csv('data/bulb_specter.csv')['n']
        return i / bulb_specter

    data = pd.DataFrame()
    y_label = ''
    x_label = '$\\alpha$, $grad$'

    data[x_label] = _convert_from_alpha_to_wavelength(v_from_alpha['alpha'])
    data[y_label] = _normalize_to_bulb_specter(_convert_to_i(v_from_alpha['V'] - v_0, r))
    ax = data.plot.scatter(x=x_label, y=y_label)
    ax.set_ylim(0, None)
    ax.set_xlim(1500, 3500)

    ax.set_ylim(0, np.max(data[y_label]) * 1.1)

    return ax


def main():
    cd_s = pd.read_csv('./data/CdS.csv')
    v_0_cd_s = 0.18
    ax = plot_specter(v_from_alpha=cd_s, v_0=v_0_cd_s, r=4.0e+04)

    cd_se = pd.read_csv('./data/CdSe.csv')
    v_0_cd_se = 0.31
    ax = plot_specter(cd_se, v_0_cd_se, r=4.0e+04)

if __name__ == '__main__':
    main()
