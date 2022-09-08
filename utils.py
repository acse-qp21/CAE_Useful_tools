"""
This module contains extra functions for training.py/ sfc_cae.py as supplement.
Author: Jin Yu
Github handle: acse-jy220
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
import vtk
import vtktools
import numpy as np
import time
import glob
import progressbar
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.tri as tri
import meshio
import re

# create an animation
from matplotlib import animation
from IPython.display import HTML
# custom colormap
import cmocean

import torch  # Pytorch
import torch.nn as nn  # Neural network module
import torch.nn.functional as fn  # Function module
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler, TensorDataset, Dataset


#################################################### Functions for data pre-processing / data loading ######################################################################

def read_in_files(data_path, file_format='vtu', vtu_fields=None):
    '''
    This function reads in the vtu/txt files in a {data_path} as tensors, of shape [snapshots, number of Nodes, Channels]

    Input:
    ---
    data_path: [string] the data_path which holds vtu/txt files, no other type of files are accepted!!!
    file_format: [string] 'vtu' or 'txt', the format of the file.
    vtu_fields: [list] the list of vtu_fields if read in vtu files, the last dimension of the tensor, e.g. ['Velocity', 'Pressure']

    Output:
    ---
    Case 1 - file_format='vtu': (3-tuple) [torch.FloatTensor] full_stage over times step, time along 0 axis; [torch.FloatTensor] coords of the mesh; [dictionary] cell_dict of the mesh.

    Case 2 - file_format='txt': [torch.FloatTensor] full_stage over times step, time along 0 axis

    '''
    data = glob.glob(data_path + "*")
    num_data = len(data)
    file_prefix = data[0].split('.')[-2].split('_')
    file_prefix.pop(-1)
    if len(file_prefix) != 1: file_prefix = '_'.join(file_prefix) + "_"
    else: file_prefix = file_prefix[0] + "_"
    file_format = '.' + file_format
    print('file_prefix: %s, file_format: %s' % (file_prefix, file_format))
    cnt_progress = 0
    if (file_format == ".vtu"):
        print("Read in vtu Data......\n")
        bar=progressbar.ProgressBar(maxval=num_data)
        bar.start()
        data = []
        coords = None
        cells = None
        start = 0
        while(True):
            if not os.path.exists(F'{file_prefix}%d{file_format}' % start):
                print(F'{file_prefix}%d{file_format} not exist, starting number switch to {file_prefix}%d{file_format}' % (start, start+1))
                start += 1
            else: break
        for i in range(start, num_data + start):
            data.append([])
            vtu_file = meshio.read(F'{file_prefix}%d{file_format}' % i)
            if not (coords == vtu_file.points).all():
               coords = vtu_file.points
               cells = vtu_file.cells_dict
               print('mesh adapted at snapshot %d' % i)
            for j in range(len(vtu_fields)):
                vtu_field = vtu_fields[j]
                if not vtu_field in vtu_file.point_data.keys():
                   raise ValueError(F'{vtu_field} not avaliable in {vtu_file.point_data.keys()} for {file_prefix} %d {file_format}' % i)
                field = vtu_file.point_data[vtu_field]
                if j == 0:
                   if field.ndim == 1: field = field.reshape(field.shape[0], 1)
                   data[i - start] = field
                else:
                   if field.ndim == 1: field = field.reshape(field.shape[0], 1)
                   data[i - start] = np.hstack((data[i - start], field))
            cnt_progress +=1
            bar.update(cnt_progress)
        bar.finish()
        whole_data = torch.from_numpy(np.array(data)).float()
        
        # get rid of zero components
        zero_compos = 0
        for i in range(whole_data.shape[-1]):
            if whole_data[..., i].max() - whole_data[..., i].min() < 1e-8:
               zero_compos += 1
               whole_data[..., i:-1] = whole_data[..., i + 1:]
        if zero_compos > 0 : whole_data = whole_data[..., :-zero_compos]
        
        return whole_data, coords, cells    

    elif (file_format == ".txt" or file_format == ".dat"):
        print("Read in txt/dat Data......")
        bar=progressbar.ProgressBar(maxval=num_data)
        data = []
        for i in range(num_data):
            data[i] = torch.from_numpy(np.loadtxt('{file_prefix} %d {file_format}' % i)).float()
            cnt_progress +=1
            bar.update(cnt_progress)
        bar.finish()
        return torch.cat(data, -1)