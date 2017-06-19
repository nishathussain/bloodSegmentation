import matplotlib.pyplot as plt
import numpy as np

import cv2
import os
from scipy.misc import imread, imsave
from model_WBC import  get_unet,model_mask,myGenerator,myTestGen
from keras.callbacks import ModelCheckpoint
#from keras.layers.convolutional import Conv2DTranspose
from keras.models import load_model
from tqdm import trange

model = model_mask(3,None,None)
model.load_weights("mask_model.h5") 
#model = load_model("mask_model.h5")
#model.summary()

#validation_data=myGenerator(gen_data='ag_val_constant')
test_data=myTestGen()
#validation_steps=250
test_steps = 64
#YY = model.predict_generator(validation_data, validation_steps)

#Incomplete:: Removing small false positive patches and counting.
# def validContours(Y): 
#     # Empirical constants
#     #MIN_DIFFERENCE = 30
#     MIN_AREA = 50
#     MAX_AREA = 200
#     #MAX_AVG_DIFF = 50
#     contours = cv2.findContours(Y.astype(np.uint8), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
#     valid_contours = []
#     for cnt in contours :
#         area = cv2.contourArea(cnt) 
#         if(area>MIN_AREA and area<MAX_AREA) :
#             p = Polygon(shell=cnt[:, 0, :])
#             x,y = p.centroid.coords[0]
#             x = int(x)
#             y = int(y)
#             cls = self.class_from_color(dot_img[y,x])
#             if cls is None: continue
#             #print(cls, x,y)
#             sealions.append ( (cls, x, y) )
#             #mask image where sealions are found
#             mask_img[y-d:y+d,x-d:x+d,:] = 0
#             valid_contours.append(cnt)
#         else:
#             cv2.drawContours(clone_img_diff, [cnt], -1, (0,0,0), thickness=cv2.FILLED)

if True: # test
    cnt = 0
    for v in trange(test_steps): #trange(validation_steps):
        print(v)
        for X,f in test_data:#validation_data:
            #print(X.shape)
            #X=myTestGen()
            #print('shape : ',X.shape)
            YY = model.predict(X)
            print(X.shape, YY.shape)
            for i in range(YY.shape[0]):
                X=np.transpose(X,(0,2,3,1))
                imsave("tmpTestMask/"+f, X[i])
                imsave("tmpTestMask/"+f[:-4]+'-mask.jpg', YY[i][0])
                cnt += 1


#Validation
if False:
    cnt = 0
    for v in trange(validation_steps):
        for X,Y in validation_data:
            YY = model.predict(X)
            print(X.shape, Y.shape, YY.shape)
            for i in range(Y.shape[0]):
                imsave("tmp/"+str(cnt)+"mask.png", Y[i][0])
                imsave("tmp/"+str(cnt)+"in.png", X[i][0])
                imsave("tmp/"+str(cnt)+".png", YY[i][0])
                cnt += 1
