from numpy import zeros, ones
from numpy.random import randn, randint
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, concatenate, MaxPooling2D, BatchNormalization
import numpy as np
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras import layers


def define_discriminator(input_shape):
    inputs = layers.Input(shape=input_shape)
    target = layers.Input(shape=input_shape)
    
    x = layers.Concatenate()([inputs, target])
    
    down1 = layers.Conv2D(64, (3, 3), strides=(3, 3), padding='same')(x)
    down1 = layers.LeakyReLU()(down1)
    
    down2 = layers.Conv2D(32, (3, 3), strides=(3, 3), padding='same')(down1)
    down2 = layers.BatchNormalization()(down2)
    down2 = layers.LeakyReLU()(down2)
    
    down3 = layers.Conv2D(16, (3, 3), strides=(3, 3), padding='same')(down2)
    down3 = layers.BatchNormalization()(down3)
    down3 = layers.LeakyReLU()(down3)
    
    down4 = layers.Conv2D(1, (3, 3), strides=(3, 3), padding='same')(down3)
    dense = layers.Flatten()(down4)
    dense = layers.Dense(100, activation='sigmoid')(dense)
    outputs = layers.Dense(1, activation='sigmoid')(dense)
    
    dis_mod = tf.keras.Model([inputs, target], outputs, name="Discriminator")
    
    def discriminator_loss(real_output, fake_output):
        real_loss = BinaryCrossentropy()(tf.ones_like(real_output), real_output)
        fake_loss = BinaryCrossentropy()(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    dis_mod.compile(loss=discriminator_loss, optimizer=opt, metrics=['accuracy'])
    
    return dis_mod

input_shape=(3, 110240, 1)
test_discr = define_discriminator(input_shape)
#print(test_discr.summary())




def encoder(input_shape):

    #Downsampling block 1
    x = Conv2D(16, (3, 3), padding = "same", dilation_rate=1)(input_shape)
    x = Conv2D(16, (3, 3), padding = "same", dilation_rate=1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D((1, 2))(x)
    ds1 = Dropout(0.1)(x)

    #Downsampling block 2
    x = Conv2D(32, (3, 3), padding = "same", dilation_rate=2)(ds1)
    x = Conv2D(32, (3, 3), padding = "same", dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D((1, 2))(x)
    ds2 = Dropout(0.1)(x)

    #Downsampling block 3
    x = Conv2D(64, (3, 3), padding = "same", dilation_rate=4)(ds2)
    x = Conv2D(64, (3, 3), padding = "same", dilation_rate=4)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D((1, 2))(x)
    ds3 = Dropout(0.1)(x)

    #Downsampling block 4
    x = Conv2D(128, (3, 3), padding = "same", dilation_rate=16)(ds3)
    x = Conv2D(128, (3, 3), padding = "same", dilation_rate=16)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D((1, 2))(x)
    ds4 = Dropout(0.1)(x)

    return ds4, ds3, ds2, ds1

def bottleneck(encoder_output):
    # Bottleneck layer
    x = Conv2D(256, (3, 3), padding = "same")(encoder_output)
    x = Conv2D(256, (3, 3), padding = "same")(x)
    x = BatchNormalization()(x)
    out = LeakyReLU(alpha=0.3)(x)
    
    return out

def decoder(bottleneck_output, ds4, ds3, ds2, ds1):

    #Upsampling Block 4
    x = Conv2DTranspose(128, (3, 3), (1, 1), padding = "same")(bottleneck_output)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = concatenate([x, ds4])

    x = Conv2D(128, (3, 3), padding = "same")(x)
    x = Conv2D(128, (3, 3), padding = "same")(x)
    up4 = LeakyReLU(alpha=0.3)(x)

    #Upsampling Block 3
    x = Conv2DTranspose(64, (3, 3), (1, 2), padding = "same")(up4)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = concatenate([x, ds3])

    x = Conv2D(64, (3, 3), padding = "same")(x)
    x = Conv2D(64, (3, 3), padding = "same")(x)
    up3 = LeakyReLU(alpha=0.3)(x)

    #Upsampling Block 2
    x = Conv2DTranspose(32, (3, 3), (1, 2), padding = "same")(up3)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = concatenate([x, ds2])

    x = Conv2D(32, (3, 3), padding = "same")(x)
    x = Conv2D(32, (3, 3), padding = "same")(x)
    up2 = LeakyReLU(alpha=0.3)(x)

    #Upsampling Block 1
    x = Conv2DTranspose(16, (3, 3), (1, 2), padding = "same")(up2)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = concatenate([x, ds1])

    x = Conv2D(32, (3, 3), padding = "same")(x)
    x = Conv2D(32, (3, 3), padding = "same")(x)
    up1 = LeakyReLU(alpha=0.3)(x)

    return up1

def define_generator(input_shape):
    inputs = Input(shape=input_shape)
    ds4, ds3, ds2, ds1 = encoder(inputs)
    bottleneck_output = bottleneck(ds4)
    decoder_output = decoder(bottleneck_output, ds4, ds3, ds2, ds1)
    
    x = Conv2DTranspose(1, (3, 3), (1, 2), padding="same")(decoder_output)
    outputs = LeakyReLU(alpha=0.3)(x)
    
    generator = Model(inputs, outputs, name="Generator")
    return generator

# Example usage:
input_shape = (3, 110240, 1)
test_gen = define_generator(input_shape)
#test_gen.summary()

   
def define_gan(g_model, d_model, theta1=1.0):
    d_model.trainable = False
    
    gen_inp = g_model.input
    gen_output = g_model.output
    
    gan_output = d_model([gen_output, gen_inp])
    
    model = Model(gen_inp, gan_output)
    
    
    @tf.function
    def generator_loss(y_true, y_pred): 
        #generated_output, generated_images = y_pred
        generated_output = y_pred[0] 
        generated_images = y_pred[1] 
        real_images = y_true 
        gan_loss = tf.reduce_mean(tf.math.log(generated_output)) 
        mse_loss = tf.reduce_mean(tf.square(real_images - generated_images)) 
        total_loss = -gan_loss + theta1 * mse_loss # Adversarial loss + MSE loss 
        return total_loss
       
    

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=generator_loss, optimizer=opt)
    
    return model




   
   
def generate_real_samples(dataset, n_samples):
    
    Xtrain, Ytrain = dataset
    ix = randint(0, Xtrain.shape[0], n_samples)
    XTRAIN, YTRAIN = Xtrain[ix], Ytrain[ix]
    
    # generate class labels and assign to y (don't confuse this with the above labels that correspond to cifar labels)
    y = ones((n_samples,))  #Label=1 indicating they are real
    return [XTRAIN, YTRAIN], y
   
   
def generate_fake_samples(generator, XTRAIN, n_samples):
    # predict outputs
    Y_ESTIMATED = generator.predict(XTRAIN)
    # create class labels
    y = zeros((n_samples,))  #Label=0 indicating they are fake
    return Y_ESTIMATED, y
   
def train(g_model, d_model, gan_model, dataset, n_epochs, n_batch):

    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)      
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):

            [X_real, Y_real], y_real = generate_real_samples(dataset, half_batch)
     
            d_loss_real, _ = d_model.train_on_batch([X_real, Y_real], y_real)     
            X_fake, y_fake = generate_fake_samples(g_model, X_real, half_batch)
            d_loss_fake, _ = d_model.train_on_batch([X_fake, Y_real], y_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            
            [X_real, Y_real], y_real = generate_real_samples(dataset, n_batch)
            #X_fake, y_fake = generate_fake_samples(g_model, X_real, n_batch)
            y_gan = ones((n_batch, ))
            #g_loss = gan_model.train_on_batch(X_real, y_gan)
            g_loss = gan_model.train_on_batch(X_real, [y_gan, Y_real])


            
            # Print losses on this batch
            print('Epoch:{}, Batch:{}, Batch_per_epo:{}, d1:{}, d2={}, d_avg={}\ng={}'.format(
                i+1, j+1, bat_per_epo, d_loss_real, d_loss_fake, d_loss, g_loss))
                
                
            
    # save the generator model
    g_model.save('cWGANIR_.keras')
    
    
    
d_model = define_discriminator(input_shape)
g_model = define_generator(input_shape)
gan_model = define_gan(g_model, d_model)


#################################################
#################################################


path1 = "/home/anchal/Desktop/rajesh/Clean/Dataset/MUSDBHQ/CM/XtrainCM3.npy"
path2 = "/home/anchal/Desktop/rajesh/Clean/Dataset/MUSDBHQ/CM/YtrainCM3.npy"

Xtrain =  np.random.random((10, 3, 110240, 1)) #np.load(path1)
Ytrain =  np.random.random((10, 3, 110240, 1)) # np.load(path2)
dataset = np.array([Xtrain, Ytrain])

train(g_model, d_model, gan_model, dataset, n_epochs=10, n_batch=4)

print("Model trained successfully!")