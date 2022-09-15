# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 19:32:17 2022

@author: meghar
"""
import numpy as np
from scipy.io import savemat

inp = 2
hl_1 = 4
out = 3

t1 = np.random.rand(hl_1, inp+1)
t2 = np.random.rand(out, hl_1+1)

savemat('theta.mat', mdict={'Theta1': t1, 'Theta2': t2})