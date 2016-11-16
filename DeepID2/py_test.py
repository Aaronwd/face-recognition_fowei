# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 21:29:01 2016

@author: wfw
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import myWrapper
import cv2
import retrieve_me
import skimage.io

people_name = []
face_vectors = []

#set_device
caffe.set_mode_cpu()

#load the model
net = caffe.Net('model/DeepID2_deploy.prototxt','model/model_iter_600000.caffemodel',caffe.TEST)

#load input and configure preprocessing
transformer=caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('data/mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)
net.blobs['data'].reshape(1,3,55,47)

#open camera
cap = cv2.VideoCapture(0)

while True:
    
    ret, face_img = cap.read()   
    cv2.imshow("capture", face_img)
    key = cv2.cv.WaitKey(10)
    if key == 27:  
        break 

    face_vectors = myWrapper.fromPython(face_img)
    if (face_vectors!=[]):  
        #note we can change the batch size on-the-fly
        #since we classify only one image, we change batch size from 10 to 1
        predict_label = []
        for i in range(0, len(face_vectors)):
            img = np.uint8(face_vectors[i]).reshape([55,47,3])
            image = skimage.img_as_float(img).astype(np.float32)
            net.blobs['data'].data[...] = transformer.preprocess('data', image)
        
            #compute
            out = net.forward()
            conv_feature = net.blobs['fc160'].data
            predict_label.append(retrieve_me.predict_labels(conv_feature)[0])
        
        for j  in range(0, len(predict_label)):
            if predict_label[j][1] > 0.17:
                people_name.append('我不认识你' + str(predict_label[j][1]))
            else:
                people_name.append('我认识你' + str(predict_label[j][0]) + str(predict_label[j][1]))
        print '/'.join(people_name)
        face_vectors = []
        people_name = []
    else:
        print "Nobody!!!"
cap.release()
cv2.destroyAllWindows()