"""
linear regression model

the environment variable VIZ shows prediction graph

How to run: 'VIZ=1 python3 linear_regression.py'
"""
import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1337)

class LinearRegression:
  def __init__(self, learning_rate=0.01, n_iterations=1000):
    self.learning_rate = learning_rate
    self.n_iterations = n_iterations
    self.theta = None

  def fit(self, X, y):
    m, n = X.shape

    # Adding a bias column to X
    X_b = np.hstack([np.ones((m, 1)), X])
    print(X_b)
    
    # Initializing theta with random values
    self.theta = np.random.randn(n + 1, 1)
    
    for _ in range(self.n_iterations):
      gradients = 2/m * X_b.T.dot(X_b.dot(self.theta) - y) # gradient of MSE
      self.theta -= self.learning_rate * gradients

  def predict(self, X):
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    return X_b.dot(self.theta)

if __name__ == "__main__":
  # random input data
  X = 2 * np.random.rand(100, 1)
  y = 4 + 3 * X + np.random.randn(100, 1)

  # training
  model = LinearRegression(learning_rate=0.1, n_iterations=1000)
  model.fit(X, y)

  # making prediction
  X_new = np.array([[0], [2]])
  y_pred = model.predict(X_new)
  print("\n****** PREDICTION ******")
  print(y_pred)
  print("************************\n")

  if os.getenv('VIZ') == "1":
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X_new, y_pred, color='red', linewidth=2, label='Prediction')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title('Linear Regression with Gradient Descent')
    plt.show()