# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:14:16 2022

@author: meghar
"""
from scipy.io import savemat
import numpy as np
import re
from matplotlib import pyplot as plt
fapp = 194

P = []
F = []

for i in range(1,fapp+1): 
    filename = r't1dp{}.tpl'.format(i)
    with open(filename) as f:
        lines = f.readlines()
    
    # def process_geometry(geometry, n_networks):
    #     branches = []
    #     assert "BRANCH" in geometry[0], "Does not start with a BRANCH"
    #     geometry = '\t'.join(geometry).replace('\n','\t')
    #     geometry = re.sub(r'\s+', '\t', geometry)
    #     geometry = geometry.split('BRANCH')
    #     geometry = [x for x in geometry if x != '']
    #     assert len(geometry) == n_networks, f"Only {len(geometry)}/{n_networks} found"
    #     for branch in geometry:
    #         branch = [x for x in branch.split('\t') if x != '']
    #         branch_name = branch[0]
    #         branch_length = int(branch[1])
    #         branch_values = [float(x) for x in branch[2:]]
    #         branch_x = branch_values[:branch_length+1]
    #         branch_elevation = branch_values[branch_length+1:]
    #         branches.append({
    #             "name": branch_name,
    #             "length": branch_length,
    #             "x": branch_x,
    #             "elevation": branch_elevation
    #         })
    #     return branches
      
        
    metadata = lines[:12]
    n_networks = int(lines[13])
    geometry = lines[15:24]
    len_catalog = int(lines[25])
    catalog = lines[26:26+len_catalog]
    #branches = process_geometry(geometry, n_networks)
    time_series = lines[58:]
    index = 0
    # series = []
    # while index < len(time_series):
    #     time = float(time_series[index])
    #     index += 1
    #     values = time_series[index: index+len_catalog]
    #     series.append({
    #         'time': time,
    #         'catalog': catalog,
    #         'values': [[float(y) for y in x.replace(' \n','').split(' ')] for x in values]
    #     })
    #     index += len_catalog
    pt_indices = []
    qg_indices = []
    qo_indices = []
    qw_indices = []
    
    for i, line in enumerate(catalog):
        if line[:2] == 'PT':
            print(i, line)
            pt_indices.append(i+1)
        if line[:4] == 'QGST':
            print(i, line)
            qg_indices.append(i+1)
        if line[:4] == 'QOST':
            print(i, line)
            qo_indices.append(i+1)
        if line[:4] == 'QWST':
            print(i, line)
            qw_indices.append(i+1)
    
    lasttp = lines[-1]
    datap = [0]
    for i, c in enumerate(lasttp):
        if lasttp[i] == ' ':
            datap.append(i+1)
            
    ptin = float(lasttp[datap[pt_indices[0]]:datap[pt_indices[0]+1]])
    P.append(ptin)
    ptout = float(lasttp[datap[pt_indices[1]]:datap[pt_indices[1]+1]])
    P.append(ptout)
    qgst = float(lasttp[datap[qg_indices[1]]:datap[qg_indices[1]+1]])
    F.append(qgst)
    qost = float(lasttp[datap[qo_indices[1]]:datap[qo_indices[1]+1]])
    F.append(qost)
    qwst = float(lasttp[datap[qw_indices[1]]:datap[qw_indices[1]+1]])
    F.append(qwst)

P = np.array(P).reshape((-1,2))
delP = (P[:,0] - P[:,1]).reshape((-1,1))
F = np.array(F).reshape((-1,3))

cutoff = 180
savemat('dat.mat', mdict={'X': P[:cutoff], 'Y': F[:cutoff], 'dP': delP[:cutoff]})
savemat('test.mat', mdict={'X_test': P[cutoff:], 'Y_test': F[cutoff:], 'dP_test': delP[cutoff:]})