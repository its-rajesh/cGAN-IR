from numpy import zeros, ones
from numpy.random import randn, randint
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, concatenate, MaxPooling2D, BatchNormalization
import numpy as np
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras import layers


model = tf.keras.models.load_model('/home/anchal/Desktop/rajesh/Clean/cGAN_IR/IndependentModels/models/generator_epoch_1500.h5')

opt = Adam(learning_rate=0.0002, beta_1=0.5)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


#################################################
#################################################


path1 = "/home/anchal/Desktop/rajesh/Clean/cGAN_IR/IndependentModels/saraga_vocal_test.npy"


Xtrain = np.load(path1)

print('\n\n\n')
print('Xtrain:', Xtrain.shape)
print('\n\n\n')


#Y_pred = model.predict(Xtrain[1000:1010,:,:])

#np.save('Y_pred.npy', Y_pred)
#print("Numpy Saved")

import soundfile as sf
import librosa
from matplotlib import pyplot as plt
from librosa import display

outpath = "/home/anchal/Desktop/rajesh/Clean/cGAN_IR/IndependentModels/SaragaOutputs/"
k = 400 #(532 test)
fs = 22050
Y_pred = model.predict(Xtrain[k:k+1,:,:])

print(Y_pred.shape)

generated_spectrogram_db = np.squeeze(Y_pred)
print(generated_spectrogram_db.shape)
generated_spectrogram_amp = librosa.db_to_amplitude(generated_spectrogram_db)
generated_audio = librosa.griffinlim(generated_spectrogram_amp)
sf.write(outpath+'pvocals.wav', generated_audio, fs)

plt.figure(figsize=(10, 6))
librosa.display.specshow(generated_spectrogram_db, sr=fs, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Predicted Spectrogram')
plt.show()
plt.savefig(outpath+'predicted.png')

bleed_db = librosa.db_to_amplitude(np.squeeze(Xtrain[k:k+1,:,:]))
bleed = librosa.griffinlim(bleed_db)
print(bleed_db.shape, bleed.shape)
sf.write(outpath+'bvocals.wav',bleed,fs)

plt.figure(figsize=(10, 6))
librosa.display.specshow(bleed_db, sr=fs, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Bleed Spectrogram')
plt.show()
plt.savefig(outpath+'bleed.png')




