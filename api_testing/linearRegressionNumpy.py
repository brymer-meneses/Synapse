import numpy as np


x = np.arange(1, 20, dtype=np.float)
W = np.zeros_like(x)
y = 2 * x # [2, 4, 6, ... 10]

def MSE(y, y_predicted):
    return ((y_predicted - y)**2).mean()

def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted - y).mean()

lr = 0.00001
maxEpochs = 110

for epoch in range(maxEpochs):
    y_pred = W * x
    loss = MSE(y, y_pred)
    dw = gradient(x, y, y_pred)

    W = W - lr * dw
    
    print(loss)
