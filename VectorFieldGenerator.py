#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 18:47:26 2018

@author: justinowen

Updated: Nov. 10, 2018 22:12:45 Peng Wei
Creates the div_free vector fields and save to csv file,
with coefficients uniformly distributed in [0,1].


"""
import matplotlib.pyplot as plt
import numpy as np
import random as rd

num_division = 64;
x = np.linspace(-1,1,num_division)
y = np.linspace(-1,1,num_division)
X, Y = np.meshgrid(x,y)

import csv


# generate the div free vector fields
for i in range(10):
    file_name = "./data/div_free_" + str(i).zfill(4) + ".csv"
    print(file_name)
    
#    rd.seed(datetime.now())
    
    Vx =  rd.random()*np.cos(2*np.pi*Y) + rd.random()*Y \
        + rd.random()*np.sin(np.pi*Y) + rd.random()*np.exp(Y)/np.e
    Vy =  rd.random()*np.cos(2*np.pi*X) + rd.random()*X \
        + rd.random()*np.sin(np.pi*X) + rd.random()*np.exp(X)/np.e
    with open(file_name,"w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(Vx)
        csvWriter.writerows("\n")
        csvWriter.writerows(Vy)

    plt.figure()
    plt.title("pivot='mid'; every third arrow; units='inches'")
    Q = plt.quiver(X[::3, ::3], Y[::3, ::3], Vx[::3, ::3], Vy[::3, ::3],
                   pivot='mid', units='inches')
    qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                       coordinates='figure')
    #plt.scatter(X[::3, ::3], Y[::3, ::3], color='r', s=5)
    
    plt.show()

# generates the non div free vector fields    
for i in range(10):
    file_name = "./data/non_div_free_" + str(i).zfill(4) + ".csv"
    print(file_name)
        
    Vx =  rd.uniform(-1,1)*np.cos(np.pi*Y*X) + rd.uniform(-1,1)*np.sin(np.pi*Y*X) \
        + rd.uniform(-1,1)*X*Y + rd.uniform(-1,1)*np.exp(X*Y/2)/np.e \
        + rd.uniform(-1,1)*np.exp((X-Y)/2)/np.e
                
    Vy =  rd.uniform(-1,1)*np.cos(np.pi*Y*X) + rd.uniform(-1,1)*np.sin(np.pi*Y*X) \
        + rd.uniform(-1,1)*X*Y + rd.uniform(-1,1)*np.exp(X*Y/2)/np.e \
        + rd.uniform(-1,1)*np.exp((X-Y)/2)/np.e
        
    with open(file_name,"w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(Vx)
        csvWriter.writerows("\n")
        csvWriter.writerows(Vy)
        
    plt.figure()
    plt.title("pivot='mid'; every third arrow; units='inches'")
    Q = plt.quiver(X[::3, ::3], Y[::3, ::3], Vx[::3, ::3], Vy[::3, ::3],
                   pivot='mid', units='inches')
    qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                       coordinates='figure')
    #plt.scatter(X[::3, ::3], Y[::3, ::3], color='r', s=5)
    
    plt.show()    
        
        

# test: load data
print("TESTING...")
for i in range(10):
    file_name = "./data/div_free_" + str(i).zfill(4) + ".csv"   
    print(file_name)

    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        idx = 0
        VVx = np.zeros((num_division,num_division))
        VVy = np.zeros((num_division,num_division))       
        isVx = True

        for row in reader:
            if row == ['\n']:
                isVx = False
                idx = 0
                continue
                
            if isVx==True:
                VVx[idx,:] = [float(x) for x in row]
                idx = idx + 1
            else:
                VVy[idx,:] = [float(x) for x in row]
                idx = idx + 1
        
    plt.figure()
    plt.title("pivot='mid'; every third arrow; units='inches'")
    Q = plt.quiver(X[::3, ::3], Y[::3, ::3], VVx[::3, ::3], VVy[::3, ::3],
                   pivot='mid', units='inches')
    qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                       coordinates='figure')
    #plt.scatter(X[::3, ::3], Y[::3, ::3], color='r', s=5)
    
    plt.show()
