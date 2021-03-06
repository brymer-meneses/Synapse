# Fixes import errors
import sys
sys.path.append("../")

import synapse as sn
from synapse import Tensor 
from synapse.nn.layers import Linear
from synapse.nn.activations import ReLU, Tanh
from synapse.nn.optimizers import SGD, GDM
from synapse.nn.loss import MSE
from synapse.nn.data import BatchIterator
from synapse.debugging import show_parents

import numpy as np
import pandas as pd 
train_data = pd.read_csv('./mnist/data/train.csv')
test_data = pd.read_csv('./mnist/data/test.csv')

train_data = train_data.sample(frac=1)
x_train = np.array( train_data.loc[:, 'pixel0': 'pixel783'] ) / 255
y_train = np.array( pd.get_dummies( train_data['label'] ) )

print(x_train.shape)
print(y_train.shape)


class NeuralNet(sn.nn.Model):
    def __init__(self):
        self.linear1 = Linear(784, 256)
        self.linear2 = Linear(256, 128)
        self.linear3 = Linear(128, 64)
        self.linear4 = Linear(64, 10)

    def forward(self, x):
        outL1 = ReLU(self.linear1(x))
        outL2 = ReLU(self.linear2(outL1))
        outL3 = ReLU(self.linear3(outL2))
        outL4 = ReLU(self.linear4(outL3))
        final = Tanh(outL4)
        return final

LR = 0.01
BETA = 0.7
ALPHA = 0.8
MAX_EPOCHS = 100
#optim = GDM(LR, BETA, ALPHA)
optim = SGD(LR)

model = NeuralNet()
model.compile(optim, MSE)
model.summary()

batch_iterator = BatchIterator(128)

for epoch in range(MAX_EPOCHS):
    epoch_loss = 0
    for count, (inputs, targets) in enumerate(batch_iterator(x_train, y_train)):
        
        output = model(inputs)
        loss = MSE(output, targets)
        loss.backward()

        model.optimize()
        model.zero_grad()
            
        if count % 100 == 0:
            print("\tloss: ",loss.data)
        epoch_loss = epoch_loss + loss.data
            
    print(f"\n=> Epoch: {epoch}, loss: {epoch_loss}")


    

        