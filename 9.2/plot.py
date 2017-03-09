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
def plot(data: pd.DataFrame):

    x_label = 'V, В'
    y_label = 'd, мкм'
    new_data = pd.DataFrame()
    new_data[x_label] = data['V']
    new_data[y_label] = data['d']

    ax = new_data.plot.scatter(x=x_label, y=y_label)
    return ax


def main():
    data = pd.read_csv('./data/data.csv')
    plot(data)

if __name__ == '__main__':
    main()
