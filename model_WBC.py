from keras.layers.convolutional import Conv2DTranspose
from keras.layers import Input, Conv2D, MaxPooling2D, merge

#define model
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Permute, BatchNormalization, Activation, UpSampling2D
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, Deconvolution2D, Cropping2D, AtrousConv2D
import os
from scipy.misc import imread
import numpy as np


from keras.layers.merge import Concatenate
from keras.optimizers import Adam
import keras.backend as K


def conv_bn_relu(input_layer, filters, kernel_size, 
                 activation="relu", border_mode='same', atrous_rate = 1):
    out = Conv2D(filters, (kernel_size, kernel_size), padding=border_mode, 
                       dilation_rate=(atrous_rate, atrous_rate))(input_layer)
    if activation is not None:
        out = BatchNormalization(axis=1)(out)
        out = Activation(activation)(out)
    return out

def model_mask(C,H,W):
    #C,H,W = 3,128,128
    atrous_rate = 1
    inp = Input(shape=(C,H,W))

    border_mode = 'same'

    out1 = conv_bn_relu(inp, 32, 5, border_mode=border_mode)
    out1 = conv_bn_relu(out1, 32, 5, border_mode=border_mode, activation=None)
    out11 = Activation('relu')(out1)
    out2 = MaxPooling2D((2, 2), padding='same')(out11)
    out3 = conv_bn_relu(out2, 64, 3, border_mode=border_mode, atrous_rate = atrous_rate)
    out3 = conv_bn_relu(out3, 64, 3, border_mode=border_mode, atrous_rate = atrous_rate, activation=None)
    out33 = Activation('relu')(out3)
    out4 = MaxPooling2D((2, 2), padding='same')(out33)
    out5 = conv_bn_relu(out4, 128, 3, border_mode=border_mode, atrous_rate = atrous_rate)
    out5 = conv_bn_relu(out5, 128, 3, border_mode=border_mode, atrous_rate = atrous_rate, activation=None)
    out55 = Activation('relu')(out5)
    out = MaxPooling2D((2, 2), padding='valid')(out55)

    out = conv_bn_relu(out, 128, 3, border_mode=border_mode)
    out = conv_bn_relu(out, 128, 3, border_mode=border_mode)
    out = UpSampling2D((2, 2))(out)
    out = conv_bn_relu(out, 64, 3, border_mode='same', atrous_rate = atrous_rate)
    out = conv_bn_relu(out, 64, 3, border_mode='same', atrous_rate = atrous_rate)
    out = UpSampling2D((2, 2))(out)
    out = conv_bn_relu(out, 32, 3, border_mode=border_mode, atrous_rate = atrous_rate)
    out = conv_bn_relu(out, 32, 3, border_mode=border_mode, atrous_rate = atrous_rate)
    out = UpSampling2D((2, 2))(out)
    out = conv_bn_relu(out, 1, 3, activation=None)
    out = Activation('sigmoid')(out)

    return Model(inp, out)

def myTestGen(gen_data='Test_Data',batch=1):
    Train_list= os.listdir(gen_data)
    X=[]
    count=1
    for f in Train_list:
        count+=1
        #print([train_data+os.sep+f])
        X.append(imread(gen_data+os.sep+f)/255)
        if count >batch:
            X1=X;X=[];count=1
            X1=np.transpose(X1,(0,3,1,2))
            yield X1, f


def myGenerator(gen_data='ag_train_constant',batch=15):
    #loading data
    #train_data='ag_train_constant'
    #val_data ='ag_val_constant'
    Train_list= os.listdir(gen_data)
    X=[]
    Y=[]
    count=1
    while True:
        for f in Train_list:
            if not f.startswith('msk'):
                count+=1
                #print([train_data+os.sep+f])
                X.append(imread(gen_data+os.sep+f)/255)
                Y.append(imread(gen_data+os.sep+'msk'+f[3:])/255)
                if count >batch:
                    X1=X;Y1=Y;X=[];Y=[];count=1
                    X1=np.transpose(X1,(0,3,1,2))
                    Y1=np.expand_dims(Y1,axis=1)
                    yield X1,Y1




smooth=1.0

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet(img_rows,img_cols):
    inputs = Input((3, img_rows, img_cols))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    
    #up6 = merge([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], mode='concat', concat_axis=1)
    #up6 = Concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    up6 = Concatenate(axis=1)([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4])
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Concatenate(axis=1)([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3])
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Concatenate(axis=1)([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Concatenate(axis=1)([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model