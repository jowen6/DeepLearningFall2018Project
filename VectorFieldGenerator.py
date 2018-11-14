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
<<<<<<< HEAD
import random
import pickle
import csv

n = 64
x = np.linspace(-1,1,n)
y = np.linspace(-1,1,n)
X, Y = np.meshgrid(x,y)



for i in range(2):
    file_name = "./data/div_free_" + str(i).zfill(4) + ".csv"
s = np.random.uniform(0,3,8)
Vx = s[0]*np.cos(np.pi*Y) + s[1]*Y + s[2]*np.sin(np.pi*Y) + s[3]*np.exp(Y)/np.e
Vy = s[4]*np.cos(np.pi*X) + s[5]*X + s[6]*np.sin(np.pi*X) + s[7]*np.exp(X)/np.e
   #with open('./data/div_free_' + str(i).zfill(4) + '.txt', 'wb') as fp:
   #    pickle.dump(X, fp)
   #    pickle.dump(Y, fp)
   #    pickle.dump(Vx, fp)
   #    pickle.dump(Vy, fp)

#with open ('./data/div_free_0000.txt', 'rb') as fp:
#    matrix = pickle.load(fp)
#import pdb; pdb.set_trace()
with open(file_name,"w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(Vx)
    csvWriter.writerows("\n")
    csvWriter.writerows(Vy)



=======
import pandas as pd
import random as rd
from datetime import datetime

num_division = 64;
x = np.linspace(-1,1,num_division)
y = np.linspace(-1,1,num_division)
X, Y = np.meshgrid(x,y)
>>>>>>> origin/VectorFields

#Vx = np.cos(5*Y*Y)#2*np.sin(5*Y)
#Vy = np.sin(5*X)
#
#plt.figure()
#plt.title("pivot='mid'; every third arrow; units='inches'")
#Q = plt.quiver(X[::3, ::3], Y[::3, ::3], Vx[::3, ::3], Vy[::3, ::3],
#               pivot='mid', units='inches')
#qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
#                   coordinates='figure')
##plt.scatter(X[::3, ::3], Y[::3, ::3], color='r', s=5)
#
#plt.show()

import csv


# generate the div free vector fields
for i in range(10):
    file_name = "./data/div_free_" + str(i).zfill(4) + ".csv"
    print(file_name)
    
#    rd.seed(datetime.now())
    
#    coeff = np.random.rand(6)
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
    
#    rd.seed(datetime.now())
   
#    Vx =  np.cos(np.pi*Y*X) + np.sin(np.pi*Y*X) \
#        + X*Y + np.exp(X*Y)/np.e \
#        + np.exp(X-Y)/np.e
#    Vy =  np.cos(np.pi*Y*X) + np.sin(np.pi*Y*X) \
#        + X*Y + rd.random()*np.exp(X*Y)/np.e \
#        + 0*np.exp(X-Y)/np.e
        
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
        
#    plt.figure()
#    plt.title("pivot='mid'; every third arrow; units='inches'")
#    Q = plt.quiver(X[::3, ::3], Y[::3, ::3], VVx[::3, ::3], VVy[::3, ::3],
#                   pivot='mid', units='inches')
#    qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
#                       coordinates='figure')
#    #plt.scatter(X[::3, ::3], Y[::3, ::3], color='r', s=5)
#    
#    plt.show()
        
"""
for i in range(2):
    for j in range(2):
        Vx = np.sin(5*i*Y)*Y
        Vy = np.sin(5*j*X)
        plt.figure()
        plt.title("pivot='mid'; every third arrow; units='inches'")
        Q = plt.quiver(X[::3, ::3], Y[::3, ::3], Vx[::3, ::3], Vy[::3, ::3])
               #pivot='mid', units='inches')
        #qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   #coordinates='figure')
""" 
#Various ways of plotting vector fields       
"""
plt.figure()
plt.title('Arrows scale with plot width, not view')
Q = plt.quiver(X, Y, Vx, Vy, units='width')
qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
"""

<<<<<<< HEAD
plt.figure()
#plt.title("pivot='mid'; every third arrow; units='inches'")
Q = plt.quiver(X[::3, ::3], Y[::3, ::3], Vx[::3, ::3], Vy[::3, ::3],
               pivot='mid', units='inches')
qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
#plt.scatter(X[::3, ::3], Y[::3, ::3], color='r', s=5)
=======

>>>>>>> origin/VectorFields


"""
plt.figure()
plt.title("pivot='tip'; scales with x view")
M = np.hypot(Vx, Vy)
Q = plt.quiver(X, Y, Vx, Vy, M, units='x', pivot='tip', width=0.022,
               scale=1 / 0.15)
qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
plt.scatter(X, Y, color='k', s=5)
"""
<<<<<<< HEAD
#plt.show()
=======
>>>>>>> origin/VectorFields
