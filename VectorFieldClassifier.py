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
import numpy as np

#Number of each type of vector field (make divisible by 2)
NumSimulations = 10
NumTestVecFields = 60   #Will create twice this number of samples
LearningRate = 0.01



def my_classifier(NumTrainVecFields, isEnriched):
    
    TestData, TestDataClassification = vf.GenerateFieldDataset(NumTestVecFields)
    secondTestData, secondTestDataClassification = vf.secondGenerateFieldDataset(NumTestVecFields)
    
    if isEnriched == False:
        #Saving Data
        vf.SaveFieldDataset(TestData,"./data/TestDataset_1.txt")
        vf.SaveFieldDataset(TestDataClassification,"./data/TestClassification_1.txt")
    else:
        # comment the above two lines if want to use the second set test data
        vf.SaveFieldDataset(secondTestData,"./data/TestDataset_1.txt")
        vf.SaveFieldDataset(secondTestDataClassification,"./data/TestClassification_1.txt")
    
    confidenfe_rate = np.zeros(4)
    average_success_rate = 0
    
    for idx in range(NumSimulations):
        print("==================================================== \n")
        
        #Generating vector fields
        TrainData, TrainDataClassification = vf.GenerateFieldDataset(NumTrainVecFields)
        
        
        #Saving Data
        vf.SaveFieldDataset(TrainData,"./data/TrainDataset_1.txt")
        vf.SaveFieldDataset(TrainDataClassification,"./data/TrainClassification_1.txt")
        
        #Transform to torch tensors
        tensor_TrainData = torch.stack([torch.Tensor(i) for i in TrainData]) 
        #torch.LongTensor(TrainDataClassification)
        tensor_TrainDataClassification = torch.stack([torch.LongTensor(i) for i in TrainDataClassification])
        
        FieldTrainDataset = utils.TensorDataset(tensor_TrainData, tensor_TrainDataClassification.view(-1)) 
        FieldTrainDataloader = utils.DataLoader(FieldTrainDataset, batch_size=4, shuffle=True, num_workers=2) 

        
        tensor_TestData = torch.stack([torch.Tensor(i) for i in TestData]) 
        #torch.LongTensor(TestDataClassification)
        tensor_TestDataClassification = torch.stack([torch.LongTensor(i) for i in TestDataClassification])
        
        #Create dataset
        FieldTestDataset = utils.TensorDataset(tensor_TestData, tensor_TestDataClassification.view(-1)) 
        
        # create dataloader
        FieldTestDataloader = utils.DataLoader(FieldTestDataset, batch_size=4, shuffle=False, num_workers=2)
        
        
#        classes = ('DivFree', 'NotDivFree')
        
        
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
                self.fc3 = nn.Linear(84, 2)    #84 inputs, 2 outputs
        
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))    #Convolution->ReLU->Pooling
                x = self.pool(F.relu(self.conv2(x)))    #Convolution->ReLU->Pooling
                #print(x.size())
                x = x.view(-1, 16 * 13 * 13)  #Convert all 2D arrays to 1D arrays
                x = F.relu(self.fc1(x))     #New layer
                x = F.relu(self.fc2(x))     #New layer
                x = self.fc3(x)             #Output layer
                x = F.softmax(x, dim=1)
                return x
        
        
        net = Net()
        
        ########################################################################
        # 3. Define a Loss function and optimizer  <3<3<3
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Classification Cross-Entropy loss and SGD with momentum are the standard for
        # classifiaction problems.
        
        import torch.optim as optim
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=LearningRate, momentum=0.9)
        
        ########################################################################
        # 4. Train the network  <3<3<3
        # ^^^^^^^^^^^^^^^^^^^^
        #
        # This is when things start to get interesting.
        # We simply have to loop over our data iterator, and feed the inputs to the
        # network and optimize.
        
        for epoch in range(2):  # loop over the dataset multiple times
        
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
        
                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        print('Finished Training')
        
        ########################################################################
        # 5. Test the network   <3<3<3
        # ^^^^^^^^^^^^^^^^^^^^
        # Let's look at how the network performs on the test dataset.
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data in FieldTestDataloader:
                inputs, labels = data
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        success_rate = correct / total
        print('Accuracy of the network on the test Fields: %d %%' % (
            100 * success_rate))
        average_success_rate += success_rate/NumSimulations
        confidenfe_rate += (outputs[:,0]).numpy()/NumSimulations
                
#        with open(filename,"a+") as my_csv:
#            my_csv.write(str(success_rate)+", ")
#            my_csv.write(str((outputs[:,0]).numpy())+"\n")
        ########################################################################


    with open(("./results/success_rate_"+str(isEnriched)
    +"_lr_"+str(LearningRate)+".csv"),"a+") as write_data:
        write_data.write(str(NumTrainVecFields) + ", " 
                         + str(average_success_rate) + "\n") 
           
    with open(("./results/confidence_rate_"+str(isEnriched)
    +"_lr_"+str(LearningRate)+".csv"),"a+") as write_data:
        write_data.write(str(NumTrainVecFields) + ", " 
                         + str(confidenfe_rate[0])+ ", " 
                         + str(confidenfe_rate[1])+ ", " 
                         + str(confidenfe_rate[2])+ ", " 
                         + str(confidenfe_rate[3]) + "\n")      
        
        
