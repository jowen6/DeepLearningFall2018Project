#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 22:10:57 2018

@author: justinowen
"""

import VFields as vf

NumVecFields = 100

#Generating vector fields
Data, Classification = vf.GenerateFieldDataset(NumVecFields)

#Saving Data
vf.SaveFieldDataset(Data,"Dataset_1.txt")
vf.SaveFieldDataset(Classification,"Classification_1.txt")

#Deleting data
del Data, Classification

#Loading saved data
Data1 = vf.LoadFieldDataset("Dataset_1.txt")
Classification1 = vf.LoadFieldDataset("Classification_1.txt")

#Plot an example from the dataset
vf.PlotField(Data1[100])


