#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 21:41:56 2018

@author: justinowen

Module for vector fields
"""

import matplotlib.pyplot as plt
import numpy as np
import random as rd
import csv

# Create 64 x 64 grid on [-1,1]x[-1,1] for vector field data
num_division = 64;
x = np.linspace(-1,1,num_division)
y = np.linspace(-1,1,num_division)
X, Y = np.meshgrid(x,y)



# Makes the div free vector fields and saves them
def DivFreeMake(NumberOfFields):
    print("Writing...")
    for i in range(NumberOfFields):
        file_name = "./data/div_free_" + str(i).zfill(4) + ".csv"
        print(file_name)
        
        # x field components with random coefficients
        Vx =  rd.random()*np.cos(2*np.pi*Y) + rd.random()*Y \
            + rd.random()*np.sin(np.pi*Y) + rd.random()*np.exp(Y)/np.e
        
        # y field components with random coefficients   
        Vy =  rd.random()*np.cos(2*np.pi*X) + rd.random()*X \
            + rd.random()*np.sin(np.pi*X) + rd.random()*np.exp(X)/np.e
        
        # write field to csv file
        with open(file_name,"w+") as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(Vx)
            csvWriter.writerows("\n")
            csvWriter.writerows(Vy)


# Makes the non div free vector fields and saves them  
def NonDivFreeMake(NumberOfFields):
    print("Writing...")
    for i in range(NumberOfFields):
        file_name = "./data/non_div_free_" + str(i).zfill(4) + ".csv"
        print(file_name)
        
        # x field components with random coefficients    
        Vx =  rd.uniform(-1,1)*np.cos(np.pi*Y*X) + rd.uniform(-1,1)*np.sin(np.pi*Y*X) \
            + rd.uniform(-1,1)*X*Y + rd.uniform(-1,1)*np.exp(X*Y/2)/np.e \
            + rd.uniform(-1,1)*np.exp((X-Y)/2)/np.e
            
        # y field components with random coefficients              
        Vy =  rd.uniform(-1,1)*np.cos(np.pi*Y*X) + rd.uniform(-1,1)*np.sin(np.pi*Y*X) \
            + rd.uniform(-1,1)*X*Y + rd.uniform(-1,1)*np.exp(X*Y/2)/np.e \
            + rd.uniform(-1,1)*np.exp((X-Y)/2)/np.e
        
        # write field to csv file    
        with open(file_name,"w+") as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(Vx)
            csvWriter.writerows("\n")
            csvWriter.writerows(Vy)


# Loads data based onfile name and number. Name should be "div_free" or "non_div_free"
def LoadFieldData(Name, Number):
    print("Loading...")
    
    file_name = "./data/" + Name + "_" + str(Number).zfill(4) + ".csv"   
    print(file_name)

    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        idx = 0
        Vx = np.zeros((num_division,num_division))
        Vy = np.zeros((num_division,num_division))       
        isVx = True

        #Parse through data in Peng's format
        for row in reader:
            if row == ['\n']:
                isVx = False
                idx = 0
                continue
                
            if isVx==True:
                Vx[idx,:] = [float(x) for x in row]
                idx = idx + 1
            else:
                Vy[idx,:] = [float(x) for x in row]
                idx = idx + 1

    return Vx, Vy


# Creates a vector field plot based on input data
def PlotField(Vx,Vy):
    plt.figure()

    #quiver command gives cool arrows
    plt.quiver(X[::3, ::3], Y[::3, ::3], Vx[::3, ::3], Vy[::3, ::3],
                   pivot='mid', units='inches')
    plt.show() 










    