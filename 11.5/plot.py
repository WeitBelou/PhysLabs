import pandas as pd
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv('./data/volt-amper.csv')
    data.plot.scatter(x='V', y='I')
    plt.show()


if __name__ == '__main__':
    main()
