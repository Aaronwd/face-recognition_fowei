# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 11:23:42 2016

@author: shjtdx
"""

#!/usr/bin/env python
# -*- coding:utf8 -*-

import numpy as np
from numpy.random import RandomState
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import ProjectedGradientNMF
from sklearn.decomposition import FastICA
from sklearn.decomposition import MiniBatchSparsePCA


rng = RandomState(0)

def sqrt_norm(x):
    sqrt_x = np.sqrt(np.sum(x**2, axis=1))
    sqrt_x = sqrt_x.reshape((sqrt_x.shape[0], 1))
    norm_x = x / sqrt_x
    return norm_x
    
def sqrt_norm_single(x):
    sqrt_x = np.sqrt(np.sum(x**2))
    norm_x = x / sqrt_x
    return norm_x

def norm_data(test_x, train_x):
    print 'sqrt normalizing data ...'
    norm_test  = sqrt_norm_single(test_x)
    norm_train = sqrt_norm(train_x)
    return norm_test, norm_train

def center_data(test_x, train_x):
    test_sample  = test_x.shape[0]
    train_sample = train_x.shape[0]
    all_x = np.vstack([test_x, train_x])
    all_centered = all_x - all_x.mean(axis=0)
    all_centered -= all_centered.mean(axis=1).reshape(test_sample + train_sample, -1)
    
    return all_centered[0:test_sample], all_centered[test_sample:]

def pca_data(test_x, train_x, params):
    print 'pcaing data ...'
    components = int(params['components'])
    pca = RandomizedPCA(components, whiten=True).fit(train_x)
    pca_train_x = pca.transform(train_x)
    pca_test_x  = pca.transform(test_x)
    return pca_test_x, pca_train_x

def eigen_face(test_x, train_x, params):
    print 'centering data ...'
    center_test, center_train = center_data(test_x, train_x)
    return pca_data(center_test, center_train, params)

def FastICA_data(test_x, train_x, params):
    print 'centering data ...'
    center_test, center_train = center_data(test_x, train_x)

    print 'icaing data ...'
    components = int(params['components'])
    ica = FastICA(n_components=components, whiten=True).fit(train_x)
    ica_train_x = ica.transform(train_x)
    ica_test_x  = ica.transform(test_x)
    return ica_test_x, ica_train_x

def sim_metric_cos(sample, train_x):
    return 1 - np.inner(train_x, sample)

def sim_metric_euc(sample, train_x):
    return np.sum( (train_x - sample) ** 2, axis=1)

pre_process_methods_set = {
    'pca': pca_data, 
    'None':None, 
    'eigen': eigen_face,
    'ica': FastICA_data,
}
sim_metric_methods_set  = {
    'euc': sim_metric_euc, 
    'cos': sim_metric_cos
}
