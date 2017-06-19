import matplotlib.pyplot as plt
import numpy as np

import os
from scipy.misc import imread, imsave
from model_WBC import  get_unet,model_mask,myGenerator,myTestGen
from keras.callbacks import ModelCheckpoint
#from keras.layers.convolutional import Conv2DTranspose
from keras.models import load_model
from tqdm import trange

model = get_unet(None,None)
model.load_weights("unet_model.h5") 
#model = load_model("mask_model.h5")
#model.summary()

#validation_data=myGenerator(gen_data='ag_val_constant')
test_data=myTestGen()
#validation_steps=250
test_steps = 64
#YY = model.predict_generator(validation_data, validation_steps)


if True: # test
    cnt = 0
    for v in trange(test_steps): #trange(validation_steps):
        print(v)
        for X in test_data:#validation_data:
            #print(X.shape)
            #X=myTestGen()
            print('shape : ',X.shape)
            YY = model.predict(X)
            #print(X.shape, YY.shape)
            for i in range(YY.shape[0]):
                imsave("tmpTestUnet/"+str(cnt)+"in.png", X[i][0])
                imsave("tmpTestUnet/"+str(cnt)+".png", YY[i][0])
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
