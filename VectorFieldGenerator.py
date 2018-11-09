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
Vx = 2*np.sin(5*Y)
Vy = 2*np.sin(5*X)

plt.figure()
plt.title('Arrows scale with plot width, not view')
Q = plt.quiver(X, Y, Vx, Vy, units='width')
qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')

plt.show()