import pandas as pd
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv('./data/volt-amper.csv')

    data['V'] = data['V'] * 1e-3

    ax = data.plot.scatter(x='V', y='I')

    ax.set_xlim(0)
    ax.set_ylim(0)

    ax.set_xlabel(r"V, В")
    ax.set_ylabel(r"I, мА")

    ax.grid()

    plt.show()


if __name__ == '__main__':
    main()
