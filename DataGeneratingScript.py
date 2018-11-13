#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 22:10:57 2018

@author: justinowen
"""

import VFields as vf

NumVecFields = 2

vf.DivFreeMake(NumVecFields)

vf.NonDivFreeMake(NumVecFields)

Vx,Vy = vf.LoadFieldData("div_free", 1)
vf.PlotField(Vx,Vy)

Vx,Vy = vf.LoadFieldData("non_div_free", 1)
vf.PlotField(Vx,Vy)