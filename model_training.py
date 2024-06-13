from sklearn.preprocessing import add_dummy_feature
import numpy as np

# Normal equation
# θ^ = (X⊺ X)-1 X⊺ y
# Find the value of θ that minimizes the MSE (mean square error)
np.random.seed(42)
# number of instances
m = 100
X = 2 * np.random.rand(m, 1)
# y = 4 + 3x₁ + Gaussian noise
y = 4 + 3 * X + np.random.randn(m, 1)
# Add x0 = 1 to each instance
X = add_dummy_feature(X)
# Compute the inverse of a matrix and perform matrix multiplication
theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
print('theta_hat', theta_hat)
# Predict using θ^
X_test = np.array([[0], [2]])
X_test = add_dummy_feature(X_test)
y_predict = X_test @ theta_hat
print('y_predict', y_predict)

# Find theta hat by computing the pseudoinverse using
# a standard matrix factorization technique called singular
# value decomposition (SVD). Which is more efficient than
# computing the Normal equation.
theta_hat_pinv = np.linalg.pinv(X) @ y
print('pseudoinverse', theta_hat_pinv)


# Batch Gradient Descent
# learning rate
eta = 0.1
n_epochs = 1000
# randomly initialized model parameters
theta = np.random.randn(2, 1)

for _ in range(n_epochs):
    gradients = 2 / m * X.T @ (X @ theta - y)
    theta = theta - eta * gradients

print('theta', theta)
