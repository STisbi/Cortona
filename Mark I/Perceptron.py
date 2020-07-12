import numpy as np


class Perceptron:
    mNumInputs = None
    mNumNeurons = None

    mInputs = None
    mTargets = None
    mWeights = None
    mBias = None

    def __init__(self, pNumNeurons, pNumInputs):
        self.mNumNeurons = pNumNeurons
        self.mNumInputs = pNumInputs

    def Initialize(self):
        self.InitializeWeights()
        self.InitializeBias()
        self.InitializeInputs()
        self.InitializeTargets()

    def InitializeWeights(self):
        self.mWeights = np.random.rand(self.mNumNeurons, self.mNumInputs)

        self.mWeights = np.array([1.0, -0.8])

    def InitializeBias(self):
        self.mBias = np.random.rand(self.mNumNeurons, 1)

    def InitializeInputs(self):
        self.mInputs = np.array([[1,2],[-1,2], [0,-1]]).transpose()

    def InitializeTargets(self):
        self.mTargets = np.array([1, 0, 0])

    def getInputs(self):
        return self.mInputs

    def getTargets(self):
        return self.mTargets

    def PrintAll(self):
        self.PrintWeights()
        self.PrintBias()
        self.PrintInputs()
        self.PrintTargets()

    def PrintWeights(self):
        print('Weight Shape:', self.mWeights.shape, '\n', self.mWeights, '\n')

    def PrintBias(self):
        print('Bias Shape:', self.mInputs.shape, '\n', self.mBias, '\n')

    def PrintInputs(self):
        print('Input Shape:', self.mInputs.shape, '\n', self.mInputs, '\n')

    def PrintTargets(self):
        print('Targets Shape:', self.mTargets.shape, '\n', self.mTargets, '\n')

    def Run(self):
        print('Initial Weights')
        self.PrintWeights()

        for iteration in range(self.mInputs.shape[1]):
            ithInput = np.array(self.mInputs[:,iteration])

            result_weight = np.dot(self.mWeights, ithInput)
            # result_bias = np.add(result_weight, self.mBias)

            actual = self.HardLim(result_weight)

            error = self.mTargets[iteration] - actual

            ep = np.multiply(ithInput, error)

            self.AdjustWeights(ep)

            print('Iteration: ', iteration)
            self.PrintWeights()

    def HardLim(self, pNetOutput):
        return 1 if (pNetOutput >= 0).all() else 0

    def AdjustWeights(self, pErroredInput):
        self.mWeights = np.add(self.mWeights, pErroredInput)

    def AdjustBias(self, pErrorInput):
        self.mBias = np.add(self.mBias, pErrorInput)

def main():
    # Parameters are (S, R) = (Number of Neurons, Input Size)
    perceptron = Perceptron(1, 2)

    perceptron.Initialize()
    # perceptron.PrintAll()
    perceptron.Run()


if __name__ == '__main__':
    main()