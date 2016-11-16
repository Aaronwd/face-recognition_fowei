# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 22:32:15 2016

@author: shjtdx
"""
import load_data_my
from pre_process_single import *
from parse import *
import numpy as np
import sys
import os

def load_train_and_test(train_data_folder):
    train_file_names = load_data_my.get_files(train_data_folder)
    train_x, train_y = load_data_my.load_data_xy(train_file_names)
    return train_x, train_y

def search(test_x, train_x, train_y, str_pre_process, str_sim_metric, params):
    pre_process_method = pre_process_methods_set[str_pre_process]
    if pre_process_method != None:
        test_x, train_x = pre_process_method(test_x, train_x, params)

    sim_metric_method = sim_metric_methods_set[str_sim_metric]
    if str_sim_metric == 'cos':
        test_x, train_x = norm_data(test_x, train_x)
    
    assert test_x.shape[1] == train_x.shape[1]
    query_sample_num = len(test_x)
    
    search_results = []
    for i in range(query_sample_num):
        sample = test_x[i]
        sim_result = sim_metric_method(sample, train_x)
        sort_index = np.argsort(sim_result)

        search_result = []
        for index in sort_index[0:10]:
            search_result.append((train_y[index], sim_result[index]))
        search_results.append((search_result))
    return search_results

def predict_labels(test_x):
    
    train_data_folder = 'train_vec'
    params_results = [{'pre_process_method': 'None', '#description': 'direct retrieval based on pixels with cosine', 'id': 'exp_2', 'sim_metric_method': 'cos'}]
    
    train_x, train_y = load_train_and_test(train_data_folder)
    
    predict_label = []
    for params in params_results:
        str_pre_process = params['pre_process_method']
        str_sim_metric  = params['sim_metric_method']
        search_results = search(test_x, train_x, train_y, 
                str_pre_process, str_sim_metric, params)
        #predict the result
        for line in search_results:
            predict_label.append(line[0])
    return predict_label

