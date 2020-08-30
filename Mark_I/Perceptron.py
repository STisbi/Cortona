from Data import Generate
import numpy as np


class Perceptron:
    num_inputs = None
    num_neurons = None
    num_epochs = None

    inputs = None
    targets = None
    weights = None
    bias = None
    error = 0

    def __init__(self, num_neurons, num_inputs, num_epochs):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.num_epochs = num_epochs

    def Initialize(self):
        self.InitializeWeights()
        self.InitializeBias()
        self.InitializeInputs()
        self.InitializeTargets()

    def InitializeWeights(self):
        self.setWeights(np.random.rand(self.num_neurons, self.num_inputs))

        # self.setWeights(np.array([0.5, -1, -0.5]))

    def InitializeBias(self):
        self.setBias(np.random.rand(self.num_neurons, 1))

        # self.setBias(np.array([0.5]))

    def InitializeInputs(self):
        self.setInputs(np.array([[1, -1, -1], [1, 1, -1]]).transpose())

    def InitializeTargets(self):
        self.setTargets(np.array([0, 1]))

    # The matrix is always written as a (S, R)
    def setWeights(self, weights):
        self.weights = weights

        if self.weights.ndim < 2:
            self.weights = self.weights.reshape((1, self.weights.size))

    # The matrix is always written as a (S, R)
    def setBias(self, bias):
        self.bias = bias

        if self.bias.ndim < 2:
            self.bias = self.bias.reshape((1, self.bias.size))

    def setInputs(self, inputs):
        self.inputs = inputs

        # If its a (n,) then transform it into a (n, 1)
        if inputs.ndim < 2:
            self.inputs = self.inputs.reshape((self.inputs.size, 1))

    def setTargets(self, targets):
        self.targets = targets

        # If its a (n,) then transform it into a (n, 1)
        if targets.ndim < 2:
            self.targets = self.targets.reshape((self.targets.size, 1))

    def getWeights(self):
        return self.weights

    def getBias(self):
        return self.bias

    def PrintAll(self):
        self.PrintWeights()
        self.PrintBias()
        self.PrintInputs()
        self.PrintTargets()

    def PrintWeights(self):
        print('Weight Shape:', self.weights.shape, '\n', self.weights, '\n')

    def PrintBias(self):
        print('Bias Shape:', self.bias.shape, '\n', self.bias, '\n')

    def PrintInputs(self):
        print('Input Shape:', self.inputs.shape, '\n', self.inputs, '\n')

    def PrintTargets(self):
        print('Targets Shape:', self.targets.shape, '\n', self.targets, '\n')

    def PrintError(self):
        print('Percent Error: ', ((100 * self.error) / self.inputs.shape[1]), '%')

    def Train(self):
        for epoch in range(self.num_epochs):
            for iteration in range(self.inputs.shape[1]):
                # Grab the first column of input and convert it
                # from (n,) to (n, 1)
                ith_input = np.array(self.inputs[:, iteration])
                ith_input = ith_input.reshape((ith_input.size, 1))

                result_weight = np.dot(self.weights, ith_input)
                result_bias = np.add(result_weight, self.bias)


                actual = self.HardLim(result_bias)

                error = self.targets[iteration] - actual

                ep = np.multiply(ith_input.transpose(), error)

                self.AdjustWeights(ep)
                self.AdjustBias(error)

    def Classify(self):
        for iteration in range(self.inputs.shape[1]):
            ith_input = np.array(self.inputs[:, iteration])
            ith_input = ith_input.reshape((ith_input.size, 1))

            result_weight = np.dot(self.weights, ith_input)
            result_bias = np.add(result_weight, self.bias)

            actual = self.HardLim(result_bias)

            self.CalculateError(target=self.targets[iteration], actual=actual)

            print('Target: ', self.targets[iteration], '\nResult: ', result_bias, '\n')
            print('Target: ', self.targets[iteration], '\nActual: ', actual, '\n')


    def HardLim(self, net_output):
        return 1 if (net_output >= 0).all() else 0

    def AdjustWeights(self, error_input):
        self.setWeights(np.add(self.weights, error_input))

    def AdjustBias(self, error_input):
        self.setBias(np.add(self.bias, error_input))

    def CalculateError(self, target, actual):
        if target - actual != 0:
            self.error += 1


def main():
    # Parameters are (S, R, E) = (Number of Neurons, Number of Inputs, Training Epochs)
    perceptron = Perceptron(1, 2, 10)
    generate = Generate.Generate(num_data_points=500, center_1=(10, 10), center_2=(-10, -10))

    perceptron.Initialize()
    perceptron.setInputs(generate.getCombinedTrainingData())
    perceptron.setTargets(generate.getCombinedTrainingTargets())

    perceptron.PrintAll()
    perceptron.Train()

    perceptron.setInputs(generate.getCombinedTestData())
    perceptron.setTargets(generate.getCombinedTestTargets())

    perceptron.Classify()

    perceptron.PrintError()

    perceptron.PrintWeights()
    perceptron.PrintBias()

    generate.PlotData(perceptron.getWeights(), perceptron.getBias())


if __name__ == '__main__':
    main()
