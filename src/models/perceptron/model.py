import numpy as np

# Perceptron model
class Perceptron:
    def __init__(self, input_dim, learning_rate=0.01):
        self.weights = np.zeros(input_dim)
        self.bias = 0
        self.learning_rate = learning_rate

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return step_function(z)

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            for i, x in enumerate(X):
                y_pred = self.predict(x)
                error = y[i] - y_pred
                # Update weights and bias
                self.weights += self.learning_rate * error * x
                self.bias += self.learning_rate * error
