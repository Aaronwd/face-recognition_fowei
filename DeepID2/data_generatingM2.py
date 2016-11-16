# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 21:29:01 2016

@author: wfw
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import pack_img
import myWrapper
import cv2
import skimage.io

face_vector = []
data_folder = './train_vec'
Nu = 0

caffe.set_mode_cpu()

#load the model
net = caffe.Net('model/DeepID2_deploy.prototxt'\
                ,'model/model_iter_600000.caffemodel',caffe.TEST)

#load input and configure preprocessing
transformer=caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('data/mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

face_list = np.zeros([100, 160])

cap = cv2.VideoCapture(0)
while True:
    # get a frame
    ret, frame = cap.read() 
    cv2.imshow("capture", frame)
    key = cv2.cv.WaitKey(10)  
    face_vector = myWrapper.fromPython(frame)
    if (face_vector!=[] and len(face_vector) < 2):
#        print('loading.................................')
        net.blobs['data'].reshape(1,3,55,47)
        img = np.uint8(face_vector).reshape([55,47,3])
        image = skimage.img_as_float(img).astype(np.float32)
        net.blobs['data'].data[...] = transformer.preprocess('data', image)
        
        #compute
        out = net.forward()
        conv_feature = net.blobs['fc160'].data
        
        face_list[Nu, :] = conv_feature
        Nu += 1
        if Nu == 100:
            break

print('packing.................................')
pack_img.output_data(face_list, data_folder)
print('finished.................................')
cap.release()