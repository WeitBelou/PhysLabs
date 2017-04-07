import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    plot_volt_amper()
    plot_temp()

    print(49e-6 / (300 * 6.19) * 1e8)


def plot_temp():
    data = pd.read_csv('./data/termo.csv')
    data['T'] = data['dV'] / 43
    data['P'] = 49e-6 * data['I_h'] ** 2 * 1e+3
    temp_fit = np.poly1d(np.polyfit(data['P'], data['T'], deg=1))
    print('A = {A} К * Вт'.format(A=temp_fit[1] * 1e-3))
    ax = data.plot.scatter(x='P', y='T')
    ax.plot(data['P'], temp_fit(data['P']))
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel(r"Мощность нагрева, мВт")
    ax.set_ylabel(r"Разность температур, K")
    ax.grid()
    plt.show()


def plot_volt_amper():
    data = pd.read_csv('./data/volt-amper.csv')
    vah_fit = np.polyfit(data['I'], data['V'], deg=1)
    vah_fit = np.poly1d(vah_fit)
    print('R = {R} мкОм'.format(R=vah_fit[1]))
    ax = data.plot.scatter(x='I', y='V')
    ax.plot(data['I'], vah_fit(data['I']))
    ax.set_xlabel(r"I, А")
    ax.set_ylabel(r"V, мкВ")
    ax.grid()
    plt.show()


if __name__ == '__main__':
    main()
