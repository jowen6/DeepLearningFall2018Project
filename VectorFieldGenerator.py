#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 18:47:26 2018

@author: justinowen
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1,1,50)
y = np.linspace(-1,1,50)
X, Y = np.meshgrid(x,y)
Vx = np.cos(10*Y*Y*Y)#2*np.sin(5*Y)
Vy = np.sin(5*X)

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

plt.figure()
plt.title("pivot='mid'; every third arrow; units='inches'")
Q = plt.quiver(X[::3, ::3], Y[::3, ::3], Vx[::3, ::3], Vy[::3, ::3],
               pivot='mid', units='inches')
qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
#plt.scatter(X[::3, ::3], Y[::3, ::3], color='r', s=5)


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
plt.show()