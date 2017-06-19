import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os
from scipy.misc import imread
from model_WBC import  model_mask,myGenerator
from keras.callbacks import ModelCheckpoint
#from keras.layers.convolutional import Conv2DTranspose


model =model_mask(3,128,128)
#model = load_model(args.load_model)
#model =model_mask(3,128,128)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adadelta')
checkpoint = ModelCheckpoint('mask_model.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='min')
#model.fit(X1,Y1,validation_split=0.1, epochs=30,callbacks=[checkpoint])
model.fit_generator(myGenerator(gen_data='ag_train_constant'),steps_per_epoch=1792, epochs=10, callbacks=[checkpoint], validation_data=myGenerator(gen_data='ag_val_constant'),validation_steps=250)

