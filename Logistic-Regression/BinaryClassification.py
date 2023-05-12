import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
path = 'E:\\Machine_learning\\4-Classification\\ex1data.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

# print('Data\n', data.head(10))
# print('Data Description \n', data.describe())


positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

# print('positive \n', positive)
# print('negative \n', negative)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not admitted')
ax.legend()
ax.set_xlabel('Exam 1 score')
ax.set_ylabel('Exam 2 score')
plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))                 # h(X)

nums = np.arange(-10, 10, step=1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(nums, sigmoid(nums), 'r')
plt.show()


data.insert(0, 'Ones', 1)

print('New data \n', data.head())


# set X (training data) and Y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

# print('X \n', X)
# print('y \n', y)

X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)

# print('X matrix \n', X)
# print('y matrix \n', y)
# print('theta \n', theta)


def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))             # in case of y=1
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))   # in case of y=0
    return np.sum(first - second) / (len(X))

thecost = cost(theta, X, y)
print('Cost \n', thecost)

def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameteres = int(theta.ravel().shape[1])
    grad = np.zeros(parameteres)

    error = sigmoid(X * theta.T) - y

    for i in range(parameteres):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad

import scipy.optimize as opt 
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
print('result \n', result)

CostAfterOptimize = cost(result[0], X, y)
print('Cost After Optimize \n', CostAfterOptimize)

def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if i >= 0.5 else 0 for i in probability]

theta_min = np.matrix(result[0])
prediction = predict(theta_min, X)

print('new predict \n', prediction)
print(y)

correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(prediction, y)]
accurcy = (sum(map(int, correct)) % len(correct))
print('Accurcy {0}%'.format(accurcy))