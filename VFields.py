#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 17:05:41 2018

@author: justinowen

Generates the training data set and testing data set.
Two types of testing data can be generated.
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



# Makes the div free vector fields and saves them
def DivFreeMake(NumberOfFields):
    print("Generating div free fields...")
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


# Makes the non div free vector fields and saves them
def NonDivFreeMake(NumberOfFields):
    print("Generating non div free fields...")
    NonDivFreeDataset = []
    NonDivFreeClassification = []
    for i in range(NumberOfFields*2):
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


def GenerateFieldDataset(NumberOfEachField):
    print("Generating fields data...")
    DivFreeDataset, DivFreeClassification = DivFreeMake(NumberOfEachField)
    NonDivFreeDataset, NonDivFreeClassification = NonDivFreeMake(NumberOfEachField)
    return DivFreeDataset+NonDivFreeDataset, DivFreeClassification+NonDivFreeClassification

def secondGenerateFieldDataset(NumberOfEachField):
    print("Generating second type fields data...")
    secondDivFreeDataset, DivFreeClassification = second_DivFreeMake(NumberOfEachField)
    NonDivFreeDataset, NonDivFreeClassification = NonDivFreeMake(NumberOfEachField)
    thirdDivFreeDataset, DivFreeClassification = third_DivFreeMake(NumberOfEachField)
    secondNonDivFreeDataset, NonDivFreeClassification = second_NonDivFreeMake(NumberOfEachField)
    fourthDivFreeDataset, DivFreeClassification = fourth_DivFreeMake(NumberOfEachField)
    thirdNonDivFreeDataset, NonDivFreeClassification = second_NonDivFreeMake(NumberOfEachField)
    return secondDivFreeDataset+thirdDivFreeDataset+fourthDivFreeDataset+NonDivFreeDataset+secondNonDivFreeDataset\
    +thirdNonDivFreeDataset, \
    DivFreeClassification+DivFreeClassification+DivFreeClassification\
    +NonDivFreeClassification+NonDivFreeClassification+NonDivFreeClassification


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
    # plt.quiver(X[::3, ::3], Y[::3, ::3], Vx[::3, ::3], Vy[::3, ::3],
    #                pivot='mid', units='inches')
    plt.quiver(X[::2, ::2], Y[::2, ::2], Vx[::2, ::2], Vy[::2, ::2],\
        pivot='mid', units='inches')
    plt.savefig('second_non_div_free.eps', format='eps', dpi=600)

    plt.show()
# Makes the second non div free vector fields and saves them
def second_NonDivFreeMake(NumberOfFields):
    print("Generating second non div free fields...")
    secondNonDivFreeDataset = []
    NonDivFreeClassification = []
    for i in range(NumberOfFields*2):
        # x field components with random coefficients
        Vx =  rd.uniform(-1,1)*np.cos(rd.uniform(-1,1)*np.pi*X)*Y**2 * \
        np.sin(Y) + rd.uniform(-1,1)* X**2 * Y**2
        # y field components with random coefficients
        Vy =  np.sqrt(np.exp(X*Y**2))*np.cos(X*Y)
        #Add data to NN Inputs List
        secondNonDivFreeDataset.append(np.array([Vx,Vy]))
        #Add Classification to NN Output List
        NonDivFreeClassification.append(np.array([1]))
        #NonDivFreeClassification.append(1)
    return secondNonDivFreeDataset, NonDivFreeClassification

# Makes the second div free vector fields and saves them
def second_DivFreeMake(NumberOfFields):
    print("Generating second type of div free fields...")
    secondDivFreeDataset = []
    DivFreeClassification = []
    for i in range(NumberOfFields):
        a = rd.uniform(-1,1)
        b = rd.uniform(-1,1)
        c = rd.uniform(-1,1)
        # x field components with random coefficients
        Vx =  a*X*Y**2 + b*np.cos(X)*Y + c*np.exp(-Y)*X**2
        # y field components with random coefficients
        Vy = a*(-Y**3/3.0) + b*1/2*Y**2*np.sin(X) + c*2*np.exp(-Y)*X
        #Add data to NN Inputs List
        secondDivFreeDataset.append(np.array([Vx,Vy]))
        #Add Classification to NN Output List
        DivFreeClassification.append(np.array([0]))
        #DivFreeClassification.append(0)

    return secondDivFreeDataset, DivFreeClassification
# Makes the third div free vector fields and saves them
def third_DivFreeMake(NumberOfFields):
        print("Generating third type of div free fields...")
        thirdDivFreeDataset = []
        DivFreeClassification = []
        for i in range(NumberOfFields):
            a = 4
            # x field components with random coefficients
            Vx =  rd.uniform(-1,1)*np.sin(a*Y)
            # y field components with random coefficients
            Vy = rd.uniform(-1,1)*np.cos(a*X)
            #Add data to NN Inputs List
            thirdDivFreeDataset.append(np.array([Vx,Vy]))
            #Add Classification to NN Output List
            DivFreeClassification.append(np.array([0]))
            #DivFreeClassification.append(0)

        return thirdDivFreeDataset, DivFreeClassification

# Makes the fourth div free vector fields and saves them
def fourth_DivFreeMake(NumberOfFields):
        print("Generating fourth type of div free fields...")
        fourthDivFreeDataset = []
        DivFreeClassification = []
        for i in range(NumberOfFields):
            a = 1/4
            # x field components with random coefficients
            Vx =  rd.uniform(-1,1)*np.sin(a*Y)
            # y field components with random coefficients
            Vy = rd.uniform(-1,1)*np.cos(a*X)
            #Add data to NN Inputs List
            fourthDivFreeDataset.append(np.array([Vx,Vy]))
            #Add Classification to NN Output List
            DivFreeClassification.append(np.array([0]))
            #DivFreeClassification.append(0)

        return fourthDivFreeDataset, DivFreeClassification
