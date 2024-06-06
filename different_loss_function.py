#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:37:17 2024
loss functions
@author: chengu
"""

def linear_regression_loss(X, Y, coeffs):
    predictions = np.dot(X, coeffs[1:]) + coeffs[0]
    return np.mean((Y - predictions) ** 2)

def ridge_regression_loss(X, Y, coeffs, alpha=1.0):
    predictions = np.dot(X, coeffs[1:]) + coeffs[0]
    mse = np.mean((Y - predictions) ** 2)
    l2_regularization = alpha * np.sum(coeffs[1:] ** 2)
    return mse + l2_regularization

def lasso_regression_loss(X, Y, coeffs, alpha=1.0):
    predictions = np.dot(X, coeffs[1:]) + coeffs[0]
    mse = np.mean((Y - predictions) ** 2)
    l1_regularization = alpha * np.sum(np.abs(coeffs[1:]))
    return mse + l1_regularization

def logistic_regression_loss(X, Y, coeffs):
    logits = np.dot(X, coeffs[1:]) + coeffs[0]
    prediction = 1 / (1 + np.exp(-logits))
    cost = -np.mean(Y * np.log(prediction) + (1 - Y) * np.log(1 - prediction))
    return cost

def poisson_regression_loss(X, Y, coeffs):
    linear_predictor = np.dot(X, coeffs[1:]) + coeffs[0]
    prediction = np.exp(linear_predictor)
    cost = np.mean(prediction - Y * linear_predictor)
    return cost

def huber_loss(X, Y, coeffs, delta=1.0):
    predictions = np.dot(X, coeffs[1:]) + coeffs[0]
    residuals = Y - predictions
    is_small_error = np.abs(residuals) <= delta
    squared_loss = np.square(residuals) / 2
    linear_loss = delta * (np.abs(residuals) - delta / 2)
    return np.mean(np.where(is_small_error, squared_loss, linear_loss))

def quantile_loss(X, Y, coeffs, quantile=0.5):
    predictions = np.dot(X, coeffs[1:]) + coeffs[0]
    residuals = Y - predictions
    return np.mean(np.where(residuals >= 0, quantile * residuals, (quantile - 1) * residuals))

# Load the data and create numpy arrays
data = np.genfromtxt('data/spam.csv', delimiter=',', skip_header=1)
problem2_X = data[:, :-1]  # Assuming last column is the label
problem2_Y = data[:, -1]

# Split the data into train, calibration, and test sets (40%, 20%, 40%)
n_emails = len(problem2_Y)
train_size = int(0.4 * n_emails)
calibration_size = int(0.2 * n_emails)
test_size = n_emails - train_size - calibration_size

X_train = problem2_X[:train_size]
Y_train = problem2_Y[:train_size]

X_calibration = problem2_X[train_size:train_size + calibration_size]
Y_calibration = problem2_Y[train_size:train_size + calibration_size]

X_test = problem2_X[train_size + calibration_size:]
Y_test = problem2_Y[train_size + calibration_size:]

# Example usage with Linear Regression
model = CustomModel(model_type='linear')
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

# Evaluate the model
mse = np.mean((Y_test - predictions) ** 2)
print(f"Mean Squared Error: {mse}")

# Example usage with Ridge Regression
model = CustomModel(model_type='ridge', alpha=1.0)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
mse = np.mean((Y_test - predictions) ** 2)
print(f"Mean Squared Error (Ridge): {mse}")

# Example usage with Lasso Regression
model = CustomModel(model_type='lasso', alpha=1.0)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
mse = np.mean((Y_test - predictions) ** 2)
print(f"Mean Squared Error (Lasso): {mse}")

# Example usage with Logistic Regression
model = CustomModel(model_type='logistic')
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
# For Logistic Regression, we can compute accuracy or other classification metrics
accuracy = np.mean((predictions > 0.5) == Y_test)
print(f"Accuracy (Logistic): {accuracy}")

# Example usage with Poisson Regression
model

# Load the data and create numpy arrays
data = np.genfromtxt('data/spam.csv', delimiter=',', skip_header=1)
problem2_X = data[:, :-1]  # Assuming last column is the label
problem2_Y = data[:, -1]

# Split the data into train, calibration, and test sets (40%, 20%, 40%)
n_emails = len(problem2_Y)
train_size = int(0.4 * n_emails)
calibration_size = int(0.2 * n_emails)
test_size = n_emails - train_size - calibration_size

X_train = problem2_X[:train_size]
Y_train = problem2_Y[:train_size]

X_calibration = problem2_X[train_size:train_size + calibration_size]
Y_calibration = problem2_Y[train_size:train_size + calibration_size]

X_test = problem2_X[train_size + calibration_size:]
Y_test = problem2_Y[train_size + calibration_size:]

# Example usage with Linear Regression
model = CustomModel(model_type='linear')
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

# Evaluate the model
mse = np.mean((Y_test - predictions) ** 2)
print(f"Mean Squared Error: {mse}")

# Example usage with Ridge Regression
model = CustomModel(model_type='ridge', alpha=1.0)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
mse = np.mean((Y_test - predictions) ** 2)
print(f"Mean Squared Error (Ridge): {mse}")

# Example usage with Lasso Regression
model = CustomModel(model_type='lasso', alpha=1.0)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
mse = np.mean((Y_test - predictions) ** 2)
print(f"Mean Squared Error (Lasso): {mse}")

# Example usage with Logistic Regression
model = CustomModel(model_type='logistic')
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
# For Logistic Regression, we can compute accuracy or other classification metrics
accuracy = np.mean((predictions > 0.5) == Y_test)
print(f"Accuracy (Logistic): {accuracy}")

# Example usage with Poisson Regression
model
