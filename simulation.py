#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 12:12:15 2018

@author: peng
"""

import VectorFieldClassifier as vfc

for NumTrainVecFields in range(10,500,10):
    print('Running with trainning vector field size: %d ' % 
            NumTrainVecFields)
    vfc.my_classifier(NumTrainVecFields)
