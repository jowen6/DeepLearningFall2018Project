#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 12:12:15 2018

@author: peng
The second variable in my_classifier(...,...)
is an indicator for whether using the 'secondTestData'.
True = enriched, False = not enriched.
"""

import VectorFieldClassifier as vfc
import os

NumSimulations = 1    #Total number of independent simulations
NumTestVecFields = 60   #Size of the testing set
LearningRate = 0.01    #learning rate in SDG
isEnriched = True       # second class div free + first class non div free

def clear_files(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


clear_files("./results/sr_"+str(isEnriched)
                +"_lr_"+str(LearningRate)+".csv")
clear_files("./results/cr_"+str(isEnriched)
                +"_lr_"+str(LearningRate)+".csv")

for NumTrainVecFields in range(200,420,20):
    print('Running with trainning vector field size: %d ' %
            NumTrainVecFields)
    vfc.my_classifier(NumTrainVecFields, NumTestVecFields, NumSimulations,
                  isEnriched, LearningRate)
