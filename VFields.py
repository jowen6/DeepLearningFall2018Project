#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 17:05:41 2018

@author: justinowen

This module contains the functions used to create various families of vector
fields on a 64 by 64 uniform grid of the square [-1,1]x[-1,1]. It also allows
users to save, load, and view those vector fields

The functions are:
    NonDivFreeMakeDataFamily1
    NonDivFreeMakeDataFamily2
    DivFreeMakeDataFamily1
    DivFreeMakeDataFamily2
    DivFreeMakeDataFamily3
    DivFreeMakeDataFamily4
    GenerateFieldDataset
    SaveFieldDataset
    LoadFieldDataset
    PlotField
"""

import matplotlib.pyplot as plt
import numpy as np
import random as rd
import pickle


# Create 64 x 64 grid on [-1,1]x[-1,1] for vector field data
num_division = 64;
x = np.linspace(-1,1,num_division)
y = np.linspace(-1,1,num_division)
X, Y = np.meshgrid(x,y)


    
# Makes the non div free vector fields and saves them  
def NonDivFreeMakeDataFamily1(NumberOfFields):
    #print("Generating non div free fields...")
    NonDivFreeDataset = []
    NonDivFreeClassification = []
    for i in range(NumberOfFields):
        
        # x field components with random coefficients    
        Vx =  rd.uniform(-1,1)*np.cos(np.pi*Y*X) + rd.uniform(-1,1)*np.sin(np.pi*Y*X) \
            + rd.uniform(-1,1)*X*Y + rd.uniform(-1,1)*np.exp(X*Y/2)/np.e \
            + rd.uniform(-1,1)*np.exp((X-Y)/2)/np.e
            
        # y field components with random coefficients              
        Vy =  rd.uniform(-1,1)*np.cos(np.pi*Y*X) + rd.uniform(-1,1)*np.sin(np.pi*Y*X) \
            + rd.uniform(-1,1)*X*Y + rd.uniform(-1,1)*np.exp(X*Y/2)/np.e \
            + rd.uniform(-1,1)*np.exp((X-Y)/2)/np.e
        
        #Add data to NN Inputs List
        NonDivFreeDataset.append(np.array([Vx,Vy]))
        
        #Add Classification to NN Output List
        NonDivFreeClassification.append(np.array([1])) 
        #NonDivFreeClassification.append(1)
    return NonDivFreeDataset, NonDivFreeClassification


# Makes the non div free vector fields and saves them  
def NonDivFreeMakeDataFamily2(NumberOfFields):
    #print("Generating non div free fields...")
    NonDivFreeDataset = []
    NonDivFreeClassification = []
    for i in range(NumberOfFields):
        
        # x field components with random coefficients    
        Vx = rd.uniform(-1,1)*np.cos(np.pi*X)*Y*Y*np.sin(Y) 
            
        # y field components with random coefficients              
        Vy = rd.uniform(-1,1)*np.exp((X*Y*Y/2))*np.cos(X*Y)
        
        #Add data to NN Inputs List
        NonDivFreeDataset.append(np.array([Vx,Vy]))
        
        #Add Classification to NN Output List
        NonDivFreeClassification.append(np.array([1])) 
        #NonDivFreeClassification.append(1)
    return NonDivFreeDataset, NonDivFreeClassification


# Makes the div free vector fields and saves them
def DivFreeMakeDataFamily1(NumberOfFields):
    #print("Generating div free fields...")
    DivFreeDataset = []
    DivFreeClassification = []
    for i in range(NumberOfFields):
        
        # x field components with random coefficients
        Vx =  rd.uniform(-1,1)*np.cos(2*np.pi*Y) + rd.uniform(-1,1)*Y \
            + rd.uniform(-1,1)*np.sin(np.pi*Y) + rd.uniform(-1,1)*np.exp(Y)/np.e
        
        # y field components with random coefficients   
        Vy =  rd.uniform(-1,1)*np.cos(2*np.pi*X) + rd.uniform(-1,1)*X \
            + rd.uniform(-1,1)*np.sin(np.pi*X) + rd.uniform(-1,1)*np.exp(X)/np.e
        
        #Add data to NN Inputs List
        DivFreeDataset.append(np.array([Vx,Vy]))
        
        #Add Classification to NN Output List
        DivFreeClassification.append(np.array([0])) 
        #DivFreeClassification.append(0) 
        
    return DivFreeDataset, DivFreeClassification

# Makes the div free vector fields and saves them
def DivFreeMakeDataFamily2(NumberOfFields):
    #print("Generating div free fields...")
    DivFreeDataset = []
    DivFreeClassification = []
    for i in range(NumberOfFields):
        a = rd.uniform(-1,1)
        b = rd.uniform(-1,1)
        c = rd.uniform(-1,1)
        
        # x field components with random coefficients
        Vx =  a*X*Y*Y + b*np.cos(X)*Y + c*X*X*np.exp(-Y)
        
        # y field components with random coefficients   
        Vy =  a*Y*Y*Y/3 + b*np.sin(X)*Y*Y/2 + c*2*X*np.exp(-Y)
        
        #Add data to NN Inputs List
        DivFreeDataset.append(np.array([Vx,Vy]))
        
        #Add Classification to NN Output List
        DivFreeClassification.append(np.array([0])) 
        #DivFreeClassification.append(0) 
        
    return DivFreeDataset, DivFreeClassification


# Makes the div free vector fields and saves them
def DivFreeMakeDataFamily3(NumberOfFields):
    #print("Generating div free fields...")
    #I believe the network will have a difficult time with this bc it is 
    #near the Nyquist frequency
    
    DivFreeDataset = []
    DivFreeClassification = []
    for i in range(NumberOfFields):
        
        # x field components with random coefficients
        Vx =  rd.uniform(-1,1)*np.sin(32*np.pi*Y)
        
        # y field components with random coefficients   
        Vy =  rd.uniform(-1,1)*np.cos(32*np.pi*X)
        
        #Add data to NN Inputs List
        DivFreeDataset.append(np.array([Vx,Vy]))
        
        #Add Classification to NN Output List
        DivFreeClassification.append(np.array([0])) 
        #DivFreeClassification.append(0) 
        
    return DivFreeDataset, DivFreeClassification


# Makes the div free vector fields and saves them
def DivFreeMakeDataFamily4(NumberOfFields):
    #print("Generating div free fields...")
    DivFreeDataset = []
    DivFreeClassification = []
    for i in range(NumberOfFields):
        
        # x field components with random coefficients
        Vx =  rd.uniform(-1,1)*np.sin(np.pi*Y/32)
        
        # y field components with random coefficients   
        Vy =  rd.uniform(-1,1)*np.cos(np.pi*X/32)
        
        #Add data to NN Inputs List
        DivFreeDataset.append(np.array([Vx,Vy]))
        
        #Add Classification to NN Output List
        DivFreeClassification.append(np.array([0])) 
        #DivFreeClassification.append(0) 
        
    return DivFreeDataset, DivFreeClassification


def GenerateFieldDataset(All = 0, NumberOfNonDivFreeFam1 = 0, NumberOfNonDivFreeFam2 = 0, 
                         NumberOfDivFreeFam1 = 0, NumberOfDivFreeFam2 = 0,
                         NumberOfDivFreeFam3 = 0,NumberOfDivFreeFam4 = 0):
    #User can specify exactly how many datapoints they want from each family or
    #just specify an All value that is greater than 0. This will create a dataset
    #equal numbers of representatives from all families and the number of each 
    #will be All.
    
    print("Generating fields data...")
    
    FieldDataset = []
    ClassificationDataset = []
    
    if (All > 0):
        print("     Using data from all 6 families: " + str(All) + " values each")
        NonDivFreeDataset, NonDivFreeClassification = NonDivFreeMakeDataFamily1(All)
        FieldDataset += NonDivFreeDataset
        ClassificationDataset += NonDivFreeClassification

        NonDivFreeDataset, NonDivFreeClassification = NonDivFreeMakeDataFamily2(All)
        FieldDataset += NonDivFreeDataset
        ClassificationDataset += NonDivFreeClassification
        
        DivFreeDataset, DivFreeClassification = DivFreeMakeDataFamily1(All)
        FieldDataset += DivFreeDataset
        ClassificationDataset += DivFreeClassification
        
        DivFreeDataset, DivFreeClassification = DivFreeMakeDataFamily2(All)
        FieldDataset += DivFreeDataset
        ClassificationDataset += DivFreeClassification
        
        DivFreeDataset, DivFreeClassification = DivFreeMakeDataFamily3(All)
        FieldDataset += DivFreeDataset
        ClassificationDataset += DivFreeClassification
        
        DivFreeDataset, DivFreeClassification = DivFreeMakeDataFamily4(All)
        FieldDataset += DivFreeDataset
        ClassificationDataset += DivFreeClassification
        
        
    if (NumberOfNonDivFreeFam1 > 0):
        print("     Generating from non divergence free family 1: " + str(NumberOfNonDivFreeFam1) + " values")
        NonDivFreeDataset, NonDivFreeClassification = NonDivFreeMakeDataFamily1(NumberOfNonDivFreeFam1)
        FieldDataset += NonDivFreeDataset
        ClassificationDataset += NonDivFreeClassification

    if (NumberOfNonDivFreeFam2 > 0):
        print("     Generating data from non divergence free family 2: " + str(NumberOfNonDivFreeFam2) + " values")
        NonDivFreeDataset, NonDivFreeClassification = NonDivFreeMakeDataFamily1(NumberOfNonDivFreeFam2)
        FieldDataset += NonDivFreeDataset
        ClassificationDataset += NonDivFreeClassification
        
    if (NumberOfDivFreeFam1 > 0):
        print("     Generating data from divergence free family 1: " + str(NumberOfDivFreeFam1) + " values")
        DivFreeDataset, DivFreeClassification = DivFreeMakeDataFamily1(NumberOfDivFreeFam1)
        FieldDataset += DivFreeDataset
        ClassificationDataset += DivFreeClassification
        
    if (NumberOfDivFreeFam2 > 0):
        print("     Generating data from divergence free family 2: " + str(NumberOfDivFreeFam2) + " values")
        DivFreeDataset, DivFreeClassification = DivFreeMakeDataFamily1(NumberOfDivFreeFam2)
        FieldDataset += DivFreeDataset
        ClassificationDataset += DivFreeClassification
        
    if (NumberOfDivFreeFam3 > 0):
        print("     Generating data from divergence free family 3: " + str(NumberOfDivFreeFam3) + " values")
        DivFreeDataset, DivFreeClassification = DivFreeMakeDataFamily1(NumberOfDivFreeFam3)
        FieldDataset += DivFreeDataset
        ClassificationDataset += DivFreeClassification
        
    if (NumberOfDivFreeFam4 > 0):  
        print("     Generating data from divergence free family 4: " + str(NumberOfDivFreeFam4) + " values")
        DivFreeDataset, DivFreeClassification = DivFreeMakeDataFamily1(NumberOfDivFreeFam4)
        FieldDataset += DivFreeDataset
        ClassificationDataset += DivFreeClassification
    
    return FieldDataset, ClassificationDataset


def SaveFieldDataset(Dataset,file_name):
    #file_name = "test.txt" for example
    print("Saving Data to " + file_name)
    with open(file_name, "wb") as fp:   #Pickling
        pickle.dump(Dataset, fp)
    

def LoadFieldDataset(file_name): 
    print("Loading Data from " + file_name)
    with open(file_name, "rb") as fp:   # Unpickling
        Dataset = pickle.load(fp)   

    return Dataset


# Creates a vector field plot based on input data
def PlotField(V):
    Vx = V[0]
    Vy = V[1]
    plt.figure()

    #quiver command gives cool arrows
    plt.quiver(X[::3, ::3], Y[::3, ::3], Vx[::3, ::3], Vy[::3, ::3],
                   pivot='mid', units='inches')
    plt.show() 
