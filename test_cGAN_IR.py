from numpy import zeros, ones
from numpy.random import randn, randint
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, concatenate, MaxPooling2D, BatchNormalization
import numpy as np
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras import layers


model = tf.keras.models.load_model('cGANIR.keras')

opt = Adam(learning_rate=0.0002, beta_1=0.5)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


#################################################
#################################################

path1 = "/home/anchal/Desktop/rajesh/Clean/Dataset/MUSDBHQ/CM/XtrainCM3.npy"
path2 = "/home/anchal/Desktop/rajesh/Clean/Dataset/MUSDBHQ/CM/YtrainCM3.npy"

Xtrain = np.load(path1)
Ytrain = np.load(path2) #np.random.random((10, 3, 110240, 1))
dataset = np.array([Xtrain, Ytrain])

print('\n\n\n')
print('Xtrain, Ytrain:', Xtrain.shape, Ytrain.shape)
print('\n\n\n')


Y_pred = model.predict(Xtrain[:10,:,:])

np.save('Y_pred.npy', Y_pred)
print("Numpy Saved")

import soundfile as sf

outpath = "/home/anchal/Desktop/rajesh/Clean/cGAN_IR/Test/"
t = 6
fs = 22050

pv = Y_pred[t][0]
pd = Y_pred[t][1]
po = Y_pred[t][2]
sf.write(outpath+'pvocals.wav', pv, fs)
sf.write(outpath+'pdrums.wav', pd, fs)
sf.write(outpath+'pother.wav', po, fs)

sf.write(outpath+'bvocals.wav',Xtrain[t][0],fs)
sf.write(outpath+'bdrums.wav',Xtrain[t][1],fs)
sf.write(outpath+'bother.wav',Xtrain[t][2],fs)

sf.write(outpath+'tvocals.wav',Ytrain[t][0],fs)
sf.write(outpath+'tdrums.wav',Ytrain[t][1],fs)
sf.write(outpath+'tother.wav',Ytrain[t][2],fs)


