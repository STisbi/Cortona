import numpy as np
import matplotlib.pyplot as plt

class Generate:
    data_1 = None
    data_2 = None

    def __init__(self):
        center_1 = (8, 4)
        center_2 = (-3, -2)
        distance = 10

        self.data_1 = np.random.uniform(center_1[0], center_1[1] + distance, size=(2, 100))
        self.data_2 = np.random.uniform(center_2[0], center_2[1] + distance, size=(2, 100))

    def PlotData(self):
        data_1x = np.array(self.data_1.transpose()[:, 0])
        data_1x = data_1x.reshape((data_1x.size, 1))

        data_1y = np.array(self.data_1.transpose()[:, 1])
        data_1y = data_1y.reshape((data_1y.size, 1))

        data_2x = np.array(self.data_2.transpose()[:, 0])
        data_2x = data_2x.reshape((data_2x.size, 1))

        data_2y = np.array(self.data_2.transpose()[:, 1])
        data_2y = data_2y.reshape((data_2y.size, 1))

        plt.scatter(data_1x, data_1y)
        plt.scatter(data_2x, data_2y)

        plt.show()

    def getData_1(self):
        return self.data_1

    def getData_2(self):
        return self.data_2

def main():
    generate = Generate()
    generate.PlotData()

if __name__ == '__main__':
    main()