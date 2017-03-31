import pandas as pd
import numpy as np


def get_temperature_from_v(v):
    V_0 = -70
    T_0 = 273 + 24
    alpha = 0.026
    return T_0 + (v - V_0) * alpha


def main():
    data = pd.read_csv('./data/11.3.csv')

    data['ln(R)'] = np.log(data['R'])

    data['T'] = get_temperature_from_v(data['V'])
    data['1/T'] = 1 / data['T']


    data.to_csv('./data/R(T).csv')

if __name__ == '__main__':
    main()