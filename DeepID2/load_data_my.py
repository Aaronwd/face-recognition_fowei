# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 16:10:47 2016

@author: rantai
"""

#!/usr/bin/env python
# -*- coding:utf8 -*-

import os
import sys
import numpy as np
import pickle

def get_files(vec_folder):
    file_names = os.listdir(vec_folder)
    file_names.sort()
    if not vec_folder.endswith('/'):
        vec_folder += '/'
    for i in range(len(file_names)):
        file_names[i] = vec_folder + file_names[i]
    return file_names

def load_data_xy(file_names):
    datas  = []
    labels = []
    for file_name in file_names[:-1]:
        f = open(file_name, 'rb')
        x, y = pickle.load(f)
        datas.append(x)
        labels.append(y)
    combine_d = np.vstack(datas)
    combine_l = np.vstack(labels)
    return combine_d, combine_l