#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:13:05 2018

@author: justinowen
"""

"""
Sections marked with "<3<3<3" are ones I'm happy with for now.
"""
#Need to normalize tensor data to reasonable range [-1,1].

# -*- coding: utf-8 -*-

import torch
#import torchvision
#import torchvision.transforms as transforms

########################################################################
# This section is for importing the data for training and testing. Our VFields
# module does most of the heavy lifting here.

import VFields as vf
import torch.utils.data as utils


#Generating vector fields
TrainData, TrainDataClassification = vf.GenerateFieldDataset(NumberOfNonDivFreeFam1 = 1000, 
                                                             NumberOfDivFreeFam1 = 1000)
TestData, TestDataClassification = vf.GenerateFieldDataset(NumberOfNonDivFreeFam1 = 60, 
                                                           NumberOfNonDivFreeFam2 = 60, 
                                                           NumberOfDivFreeFam1 = 60, 
                                                           NumberOfDivFreeFam2 = 60,
                                                           NumberOfDivFreeFam3 = 0,
                                                           NumberOfDivFreeFam4 = 60)


#Saving Data
vf.SaveFieldDataset(TrainData,"TrainDataset_1.txt")
vf.SaveFieldDataset(TrainDataClassification,"TrainClassification_1.txt")

vf.SaveFieldDataset(TestData,"TestDataset_1.txt")
vf.SaveFieldDataset(TestDataClassification,"TestClassification_1.txt")

"""
#Load Data if available
TrainData = vf.LoadFieldDataset("TrainDataset_1.txt")
TrainDataClassification = vf.LoadFieldDataset("TrainClassification_1.txt")

TestData = vf.LoadFieldDataset("TestDataset_1.txt")
TestDataClassification = vf.LoadFieldDataset("TestClassification_1.txt")
"""

#Transform to torch tensors
tensor_TrainData = torch.stack([torch.Tensor(i) for i in TrainData]) 
#torch.LongTensor(TrainDataClassification)
tensor_TrainDataClassification = torch.stack([torch.LongTensor(i) for i in TrainDataClassification])


tensor_TestData = torch.stack([torch.Tensor(i) for i in TestData]) 
#torch.LongTensor(TestDataClassification)
tensor_TestDataClassification = torch.stack([torch.LongTensor(i) for i in TestDataClassification])


#Create dataset
FieldTrainDataset = utils.TensorDataset(tensor_TrainData, tensor_TrainDataClassification.view(-1)) 
FieldTestDataset = utils.TensorDataset(tensor_TestData, tensor_TestDataClassification.view(-1)) 


# create dataloader
FieldTrainDataloader = utils.DataLoader(FieldTrainDataset, batch_size=4, shuffle=True, num_workers=2) 
FieldTestDataloader = utils.DataLoader(FieldTestDataset, batch_size=4, shuffle=False, num_workers=2)


classes = ('DivFree', 'NotDivFree')



########################################################################
# 2. Define a Convolution Neural Network <3<3<3
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Using the neural network from the PyTorch Neural Networks tutorial modified to
# take 2-channel images (1 channel per 2D vector component).

# I have some concerns that our vector fields may have a different variation
# of values per pixel and this may screw things up.
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 6, 5) #2 inputs, 6 outputs, 5x5 convolution window
        self.pool = nn.MaxPool2d(2, 2)  #2x2 non-overlapping window that takes max in window
        self.conv2 = nn.Conv2d(6, 16, 5)    #6 inputs, 16 outputs, 5x5 convolution window
        self.fc1 = nn.Linear(16 * 13 * 13, 120) #16*5*5 inputs, 120 outputs
        self.fc2 = nn.Linear(120, 84)   #120 inputs, 84 outputs
        self.fc3 = nn.Linear(84, 2)    #84 inputs, 10 outputs

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))    #Convolution->ReLU->Pooling
        x = self.pool(F.relu(self.conv2(x)))    #Convolution->ReLU->Pooling
        #print(x.size())
        x = x.view(-1, 16 * 13 * 13)  #Convert all 2D arrays to 1D arrays
        x = F.relu(self.fc1(x))     #New layer
        x = F.relu(self.fc2(x))     #New layer
        x = self.fc3(x)             #Output layer
        return x


net = Net()

########################################################################
# 3. Define a Loss function and optimizer  <3<3<3
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Classification Cross-Entropy loss and SGD with momentum are the standard for
# classifiaction problems.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.1)

########################################################################
# 4. Train the network  <3<3<3
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.
print('Training...')
for epoch in range(6):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(FieldTrainDataloader, 0):
        # get the inputs
        inputs, labels = data

        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


print('Finished Training')

########################################################################
# 4. Test the network   <3<3<3
# ^^^^^^^^^^^^^^^^^^^^
# Let's look at how the network performs on the test dataset.
print('Testing...')
correct = 0
total = 0
with torch.no_grad():
    for data in FieldTestDataloader:
        Testinputs, labels = data
        Testoutputs = net(Testinputs)
        #print(Testoutputs)
        _, predicted = torch.max(Testoutputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the test Fields: %d %%' % (
    100 * correct / total))
########################################################################
