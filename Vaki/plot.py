import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_vi(filename: str):
    forward_data = pd.read_csv(filename)

    plt.plot(forward_data['V'], forward_data['I'], 'o')

    plt.xlabel('V')
    plt.ylabel('I')

    plt.grid(which='both')

    plt.show()


def compute_fauler(forward_data: pd.DataFrame) -> pd.DataFrame:
    forward_data_y = np.log(forward_data['I'].values / forward_data['V'].values ** 2)
    forward_data_x = 1 / forward_data['V'].values

    data = pd.DataFrame()
    data['ln(I / U ^ 2)'] = forward_data_y
    data['1 / U'] = forward_data_x
    return data


def linear_fit():
    a = []
    lnB = []

    for i in range(1, 5):
        filename = './BIG_DATA/vi_{0}.csv'.format(i)
        data = pd.read_csv(filename)
        data = compute_fauler(data)
        x = data['ln(I / U ^ 2)'].values
        y = data['1 / U'].values

        approx = np.polyfit(x, y, 1)
        a.append(approx[0])
        lnB.append(approx[1])

    # Test
    plt.scatter(a, lnB)
    plt.show()


if __name__ == '__main__':
    linear_fit()
