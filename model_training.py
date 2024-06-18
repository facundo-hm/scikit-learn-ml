from sklearn.preprocessing import add_dummy_feature, PolynomialFeatures
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge, Lasso
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt

# Normal equation
# θ^ = (X⊺ X)-1 X⊺ y
# Find the value of θ that minimizes the MSE (mean square error)
np.random.seed(42)
# number of instances
m = 100
X_init = 2 * np.random.rand(m, 1)
# y = 4 + 3x₁ + Gaussian noise
y_init = 4 + 3 * X_init + np.random.randn(m, 1)
# Add x0 = 1 to each instance
X = add_dummy_feature(X_init)
# Compute the inverse of a matrix and perform matrix multiplication
theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y_init
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
theta_hat_pinv = np.linalg.pinv(X) @ y_init
print('pseudoinverse', theta_hat_pinv)

# Batch Gradient Descent
# learning rate
eta = 0.1
n_epochs = 1000
# randomly initialized model parameters
theta = np.random.randn(2, 1)

for _ in range(n_epochs):
    # Compute the gradient vector of the MSE cost function
    gradients = 2 / m * X.T @ (X @ theta - y_init)
    theta = theta - eta * gradients

# Stochastic Gradient Descent
n_epochs = 50
# Learning schedule hyperparameters
t0, t1 = 5, 50
# Determine the learning rate
def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for iteration in range(m):
        random_index = np.random.randint(m)
        xi = X[random_index : random_index + 1]
        yi = y_init[random_index : random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta * gradients

print('theta', theta)

# Perform linear regression using stochastic GD
sgd = SGDRegressor(
    max_iter=1000,
    tol=1e-5,
    penalty=None,
    eta0=0.01,
    n_iter_no_change=100,
    random_state=42)
sgd.fit(X_init, y_init.ravel())
print('Bias term: ', sgd.intercept_, ', Weights: ', sgd.coef_)

# Polynomial Regression
X = 6 * np.random.rand(m, 1) - 3
# Simple quadratic equation, y = ax² + bx + c plus some noise
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
# Add the square of each feature as a new feature
X_poly = poly_features.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print('Bias term: ', lin_reg.intercept_, ', Weights: ', lin_reg.coef_)

# Learning Curves
poly_reg = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False),
    LinearRegression())
# Train and evaluates the model using cross-validation to get
# an estimate of the model’s generalization performance.
train_sizes, train_scores, valid_scores = learning_curve(
    poly_reg, X_init, y, train_sizes=np.linspace(0.01, 1.0, 40),
    cv=5, scoring='neg_root_mean_squared_error')

train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

plt.plot(train_sizes, train_errors, 'r-+', linewidth=2, label='train')
plt.plot(train_sizes, valid_errors, 'b-', linewidth=3, label='valid')
plt.savefig('./charts/learning_curve')

# Ridge Regression
# Regularized version of linear regression, a regularization term
# is added to the MSE.
# Ridge regression using a closed-form solution.
ridge_cf = Ridge(alpha=0.1, solver='cholesky')
ridge_cf.fit(X_init, y_init)
ridge_cf_pred = ridge_cf.predict([[1.5]])
print('ridge_cf_pred', ridge_cf_pred)

# Ridge using stochastic gradient descent
ridge_sgd = SGDRegressor(
    penalty='l2', alpha=0.1/m, tol=None,
    max_iter=1000, eta0=0.01, random_state=42)
ridge_sgd.fit(X_init, y_init.ravel())
ridge_sgd_pred = ridge_sgd.predict([[1.5]])
print('ridge_sgd_pred', ridge_sgd_pred)

# Lasso Regression
# Regularized version of linear regression, uses the ℓ1 norm of
# the weight vector to add a regularization term to the cost function.
# Similar to SGDRegressor(penalty='l1', alpha=0.1)
lasso = Lasso(alpha=0.1)
lasso.fit(X_init, y_init)
lasso_pred = lasso.predict([[1.5]])
print('lasso_pred', lasso_pred)
