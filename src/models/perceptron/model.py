import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000, activation='sigmoid'):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.activation = activation
        self.weights = None
        self.bias = None

    '''
    The step function is challenging for gradient-based optimization (like backpropagation) since:

    Its derivative is zero almost everywhere, so it doesn't provide useful gradient information.
    The function is non-differentiable at z=0, which complicates gradient-based learning.

    Activation functions like sigmoid and ReLU are typically preferred, 
    as they provide smooth, non-zero derivatives that facilitate effective gradient-based learning.
    '''
    def _activate(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid_act(x)
        elif self.activation == 'ReLU':
            return self.ReLU_act(x)
        elif self.activation == 'step':
            return self.step_function_act(x)
        else:
            raise ValueError("Unsupported activation function")

    '''
    <activation_function>_act are used during forward feed of the neural network.
    <activation_function>_derivative are used during back propagation of the neural network.
    '''

    @staticmethod
    def sigmoid_act(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def ReLU_act(x):
        return np.maximum(0, x)

    @staticmethod
    def ReLU_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def step_function_act(z):
        return 1 if z > 0 else 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize random weights and bias in the first iteration
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for idx, sample in enumerate(X):
                # Linear combination
                linear_output = np.dot(sample, self.weights) + self.bias
                # Apply activation function
                y_predicted = self._activate(linear_output)

                # Update weights and bias
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * sample
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._activate(linear_output)
        return np.where(y_predicted >= 0.5, 1, 0)  # Threshold for binary classification

