import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os
from scipy.misc import imread
from model_WBC import  get_unet, model_mask,myGenerator, dice_coef_loss
from keras.callbacks import ModelCheckpoint
#from keras.layers.convolutional import Conv2DTranspose



model =get_unet(None,None)
model.load_weights("unet_model.h5")
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adadelta') # 'binary_crossentropy' or 'dice_coef_loss'
checkpoint = ModelCheckpoint('unet_model.h5', verbose=10, monitor='val_loss', save_best_only=True, mode='min')
#model.fit(X1,Y1,validation_split=0.1, epochs=30,callbacks=[checkpoint])
model.fit_generator(myGenerator(),steps_per_epoch=1792, epochs=1, callbacks=[checkpoint], \
                            validation_data=myGenerator(gen_data='ag_val_constant'),validation_steps=250)

