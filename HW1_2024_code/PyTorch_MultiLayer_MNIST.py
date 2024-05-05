import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable

import h5py
import time
from tqdm import tqdm

#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0]  ) )

MNIST_data.close()

#number of hidden units
H = 100

#Model architecture
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        # input is 28x28
        self.fc1 = nn.Linear(28*28, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H)
        self.fc4 = nn.Linear(H, H)
        self.fc5 = nn.Linear(H, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1( x ))
        x = F.relu(self.fc2( x ))
        x = F.relu(self.fc3( x ))
        x = F.relu(self.fc4( x ))
        x = self.fc3( x )
        
        return F.log_softmax(x, dim=1)
    

model = MnistModel()

#Stochastic gradient descent optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)


batch_size = 100
num_epochs = 100
print_freq = 5
L_Y_train = len(y_train)
model.train()
#train_loss = []

#Train Model
for epoch in range(num_epochs):
    
    #Randomly shuffle data every epoch
    I_permutation = np.random.permutation(L_Y_train)
    x_train =  x_train[I_permutation,:]
    y_train =  y_train[I_permutation] 
    train_accu = []

    for i in range(0, L_Y_train, batch_size):
        x_train_batch = torch.FloatTensor( x_train[i:i+batch_size,:] )
        y_train_batch = torch.LongTensor( y_train[i:i+batch_size] )
        #data, target = Variable(x_train_batch).cuda(), Variable(y_train_batch).cuda()
        data, target = Variable(x_train_batch), Variable(y_train_batch)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()    # calculate gradients
        #train_loss.append(loss.data)
        optimizer.step()   # update gradients
        #calculate accuracy
        prediction = output.data.max(1)[1]   # first column has actual prob.
        accuracy = ( float( prediction.eq(target.data).sum() ) /float(batch_size)  )*100.0
        train_accu.append(accuracy)
    accuracy_epoch = np.mean(train_accu)

    if (epoch % print_freq) == 0:
        print(epoch, accuracy_epoch)

print("Training is done, now perform testing.")

#Test Model
test_accu = []
for i in tqdm(range(0, len(y_test), batch_size)):
    x_test_batch = torch.FloatTensor( x_test[i:i+batch_size,:] )
    y_test_batch = torch.LongTensor( y_test[i:i+batch_size] )
    data, target = Variable(x_test_batch), Variable(y_test_batch)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)

    prediction = output.data.max(1)[1]   # first column has actual prob.
    accuracy = ( float( prediction.eq(target.data).sum() ) /float(batch_size)  )*100.0
    test_accu.append(accuracy)
    
accuracy_test = np.mean(test_accu)
print(f"The accuracy of the neural network is {accuracy_test}%.")
    
