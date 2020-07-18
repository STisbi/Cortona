import numpy as np
import matplotlib.pyplot as plt


class Generate:
    training_data_1 = None
    training_data_2 = None

    training_targets_1 = None
    training_targets_2 = None

    test_data_1 = None
    test_data_2 = None

    test_targets_1 = None
    test_targets_2 = None

    def __init__(self, num_data_points, center_1=(7,6), center_2=(-5, -6)):
        distance = 10

        self.training_data_1 = np.random.uniform(center_1[0], center_1[1] + distance, size=(2, num_data_points))
        self.training_data_2 = np.random.uniform(center_2[0], center_2[1] + distance, size=(2, num_data_points))

        # Get floor of quotient
        self.test_data_1 = np.random.uniform(center_1[0], center_1[1] + distance, size=(2, num_data_points // 4))
        self.test_data_2 = np.random.uniform(center_2[0], center_2[1] + distance, size=(2, num_data_points // 4))

        self.training_targets_1 = np.zeros(shape=(1, num_data_points))
        self.training_targets_2 = np.ones(shape=(1, num_data_points))

        self.test_targets_1 = np.zeros(shape=(1, num_data_points // 4))
        self.test_targets_2 = np.ones(shape=(1, num_data_points // 4))

    def PlotData(self):
        plt.scatter(self.training_data_1[0, :], self.training_data_1[1, :], label='Training Data 1')
        plt.scatter(self.training_data_2[0, :], self.training_data_2[1, :], label='Training Data 2')

        plt.scatter(self.test_data_1[0, :], self.test_data_1[1, :], label='Test Data 1')
        plt.scatter(self.test_data_2[0, :], self.test_data_2[1, :], label='Test Data 2')

        plt.legend()
        plt.show()

    def getTrainingData_1(self):
        return self.training_data_1

    def getTrainingData_2(self):
        return self.training_data_2

    def getTestData_1(self):
        return self.test_data_1

    def getTestData_2(self):
        return self.test_data_2

    def getTrainingTargets_1(self):
        return self.training_targets_1

    def getTrainingTargets_2(self):
        return self.training_targets_2

    def getTestTarget_1(self):
        return self.test_targets_1

    def getTestTarget_2(self):
        return self.test_targets_2

    def getCombinedTrainingData(self):
        return np.append(self.training_data_1, self.training_data_2, axis=1)

    def getCombinedTestData(self):
        return np.append(self.test_data_1, self.test_data_2, axis=1)

    def getCombinedTrainingTargets(self):
        return np.append(self.training_targets_1, self.training_targets_2)

    def getCombinedTestTargets(self):
        return np.append(self.test_targets_1, self.test_targets_2)


def main():
    generate = Generate(num_data_points=100)
    generate.PlotData()


if __name__ == '__main__':
    main()