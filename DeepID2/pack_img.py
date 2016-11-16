# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 14:53:43 2016

@author: shjtdx
"""

import os
import cPickle
import numpy as np

def cPickle_output(vars, file_name):
    f = open(file_name, 'wb')
    cPickle.dump(vars, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    
def output_data(vector_vars, vector_folder, batch_size=100):
    label = []
    if not vector_folder.endswith('/'):
        vector_folder += '/'
    if not os.path.exists(vector_folder):
        os.mkdir(vector_folder)
    if not os.path.exists(vector_folder + '/data_path.txt'):
        open(vector_folder + '/data_path.txt', "w")
    
    file_path = open(vector_folder + '/data_path.txt', "rb")
    path_name = file_path.readlines()
    if path_name is None:
        current_label = 0
    else:
        current_label = len(path_name)
    x = vector_vars
    for i in range(x.shape[0]):
        label.append((i + current_label) * np.ones([1,100], 'int32'))
    y = label
    n_batch = len(x) / batch_size
    
    with open(vector_folder + '/data_path.txt', "a") as f:
        for i in range(n_batch):
            file_name = vector_folder + str(i + current_label) + '.pkl'
            f.writelines(file_name + '\n')
            batch_x = x[ i*batch_size: (i+1)*batch_size]
            batch_y = np.array(np.matrix(y[i]).T)
            cPickle_output((batch_x, batch_y), file_name)
#        if n_batch * batch_size < len(x):
#            batch_x = x[n_batch*batch_size: ]
##        batch_y = y[n_batch*batch_size: ]
#            file_name = vector_folder + str(n_batch) + '.pkl'
#            cPickle_output((batch_x), file_name)