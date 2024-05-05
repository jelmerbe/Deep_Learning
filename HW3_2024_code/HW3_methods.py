import numpy as np
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from random import randint
import pandas as pd

# Inputing data
def load_data(batch_size=100):
    ## Defining the data augmentation transformations for the training set
    transformations = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomVerticalFlip(p=0.5), 
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train = torchvision.datasets.CIFAR10(root='./CIFARdata', train=True,
                                        download=True, transform=transformations) #data augmentation transformations
    data_train = torch.utils.data.DataLoader(train, batch_size=batch_size, #Batch size = 100
                                          shuffle=True, num_workers=0)

    transform_testset = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test = torchvision.datasets.CIFAR10(root='./CIFARdata', train=False,
                                        download=True, transform=transform_testset)
    data_test = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    return data_train, data_test   

# Define the model
class CIFAR10Model(nn.Module):
    def __init__(self, num_outputs, p, n_channels):
        super(CIFAR10Model, self).__init__() #Specifying the model parameters 
        # input is 3x32x32
        self.n_channels = n_channels
        self.p = p
        
        #8 convolution layers with ReLU activation 
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=4, stride = 1, padding = 2) 
        '''
        No. of output units in a convolution layer = 
        {[No. of input units - Filter Size + 2(Padding)]/Stride} + 1 = 
        (32 - 3 + 2(1))/1 + 1 = 32 
        '''
        self.conv_layer2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=4, stride = 1, padding = 2) 
        self.conv_layer3 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=4, stride = 1, padding = 2) 
        self.conv_layer4 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=4, stride = 1, padding = 2) 
        self.conv_layer5 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=4, stride = 1, padding = 2) 
        self.conv_layer6 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride = 1, padding = 0)     
        self.conv_layer7 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride = 1, padding = 0) 
        self.conv_layer8 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride = 1, padding = 0)  
        '''
        No. of output units in a convolution layer = 
        {[No. of input units - Filter Size + 2(Padding)]/Stride} + 1 = 
        (32 - 3 + 2(1))/1 + 1 = 32 
        '''
        #3 Fully connected layers with ReLU activation 
        self.fc1 = nn.Linear(in_features=n_channels*4*4, out_features=500) 
        # Image size is 24*24 (due to random resized cropping) with 128 channels from the last convolution layer
        self.fc2 = nn.Linear(in_features=500, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=num_outputs)    
        
        # Dropout
        self.dropout = nn.Dropout(p=self.p) # p - Probability of dropping out a neuron
        self.dropout2d = nn.Dropout2d(p=self.p)
        self.batchnorm2d_layer1 = nn.BatchNorm2d(n_channels)
        self.batchnorm2d_layer2 = nn.BatchNorm2d(n_channels) 
        self.batchnorm2d_layer3 = nn.BatchNorm2d(n_channels)
        self.batchnorm2d_layer4 = nn.BatchNorm2d(n_channels)
        self.batchnorm2d_layer5 = nn.BatchNorm2d(n_channels)
        self.pool = nn.MaxPool2d(2, stride=2) #Max pooling 

    def forward(self, x): #Specifying the NN architecture 
        x = F.relu(self.conv_layer1(x)) #Convolution layers with relu activation
        x = self.batchnorm2d_layer1(x) #Batch normalization 
        x = F.relu(self.conv_layer2(x))
        x = self.pool(x) #Pooling layer
        x = self.dropout2d(x) #Dropout
        x = F.relu(self.conv_layer3(x))
        x = self.batchnorm2d_layer2(x)
        x = F.relu(self.conv_layer4(x))
        x = self.pool(x)
        #x = self.dropout2d(x)
        x = F.relu(self.conv_layer5(x))
        x = self.batchnorm2d_layer3(x)
        x = F.relu(self.conv_layer6(x))
        #x = self.dropout2d(x)
        x = F.relu(self.conv_layer7(x))
        x = self.batchnorm2d_layer4(x)
        x = F.relu(self.conv_layer8(x)) 
        x = self.batchnorm2d_layer5(x)
        x = self.dropout2d(x)
        x = x.view(-1, self.n_channels*4*4) #Flattening the conv2D output for dropout 
        x = F.relu(self.fc1(x)) #Fully connected layer with relu activation 
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, data_train, lr, batch_size, device="cuda", num_epochs=200, print_freq=5):
    # Parameters
    loss_func = nn.CrossEntropyLoss()
    model.train() 
    train_accuracies = []
    '''
    Train Mode
    Create the batch -> Zero the gradients -> Forward Propagation -> Calculating the loss 
    -> Backpropagation -> Optimizer updating the parameters -> Prediction 
    '''
    print(f"\n\nTraining ConvNet with dropout proba={model.p}, numbers of channels={model.n_channels} and learning rate={lr}.")
    start_time = time.time()
    for epoch in range(num_epochs):
        batch_acc = []
        if epoch == 50:
            lr /= 10.0
        if epoch == 100:
            lr /= 10.0
        if epoch == 150:
            lr /= 10.0
        optimizer = optim.Adam(model.parameters(), lr=lr)
        running_loss = 0.0
        for i, batch in enumerate(data_train, 0):
            data, target = batch
            data, target = Variable(data).to(device), Variable(target).to(device)
            optimizer.zero_grad() 
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            prediction = output.data.max(1)[1]
            accuracy = (float(prediction.eq(target.data).sum())/float(batch_size))*100.0
            batch_acc.append(accuracy)   
        accuracy_epoch = np.mean(batch_acc)
        train_accuracies.append(accuracy_epoch)
        if epoch == print_freq:
            print('In epoch ', epoch,' the accuracy of the training set =', accuracy_epoch)
    end_time = time.time()
    print(f'\nTime to train the model: {(end_time-start_time) / 60.0} mins.')

    # Save trained models in correct location
    location = "/data/math-deep-learning-course/exet5913/Deep_Learning_HW3"
    torch.save(model.state_dict(), f"{location}/model_n_channels_{model.n_channels}_dropout_{model.p}_lr_{lr}")

    # torch.save(optimizer.state_dict(), f"./opt/opt_n_channels_{model.n_channels}_dropout_{model.p}_lr_{lr}")
    train_accuracies_series = pd.Series(train_accuracies)
    train_accuracies_series.to_csv(f"{location}/train_accuracies_n_channels_{model.n_channels}_dropout_{model.p}_lr_{lr}", index=False)
    return model, train_accuracies

def test(model, data_test, batch_size, device="cuda"):
    test_accuracy = []
    model.eval()
    for batch in data_test:
        data, target = batch
        data, target  = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(target.data).sum())/float(batch_size))*100.0
        test_accuracy.append(accuracy)
    print('Accuracy on the test set = ', np.mean(test_accuracy))
    return np.mean(test_accuracy)



