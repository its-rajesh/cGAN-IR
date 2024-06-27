import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np

# Define the discriminator model
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
    
    return dis_mod

# Define the encoder model
def encoder(input_shape):
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(16, (3, 3), padding='same', dilation_rate=1)(inputs)
    x = layers.Conv2D(16, (3, 3), padding='same', dilation_rate=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.MaxPooling2D((1, 2))(x)
    ds1 = layers.Dropout(0.1)(x)

    x = layers.Conv2D(32, (3, 3), padding='same', dilation_rate=2)(ds1)
    x = layers.Conv2D(32, (3, 3), padding='same', dilation_rate=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.MaxPooling2D((1, 2))(x)
    ds2 = layers.Dropout(0.1)(x)

    x = layers.Conv2D(64, (3, 3), padding='same', dilation_rate=4)(ds2)
    x = layers.Conv2D(64, (3, 3), padding='same', dilation_rate=4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.MaxPooling2D((1, 2))(x)
    ds3 = layers.Dropout(0.1)(x)

    x = layers.Conv2D(128, (3, 3), padding='same', dilation_rate=16)(ds3)
    x = layers.Conv2D(128, (3, 3), padding='same', dilation_rate=16)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.MaxPooling2D((1, 2))(x)
    ds4 = layers.Dropout(0.1)(x)

    return Model(inputs, [ds4, ds3, ds2, ds1])

# Define the bottleneck
def bottleneck(encoder_output):
    x = layers.Conv2D(256, (3, 3), padding='same')(encoder_output)
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    out = layers.LeakyReLU(alpha=0.3)(x)
    
    return out

# Define the decoder
def decoder(bottleneck_output, ds4, ds3, ds2, ds1):
    x = layers.Conv2DTranspose(128, (3, 3), (1, 1), padding='same')(bottleneck_output)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.concatenate([x, ds4])

    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    up4 = layers.LeakyReLU(alpha=0.3)(x)

    x = layers.Conv2DTranspose(64, (3, 3), (1, 2), padding='same')(up4)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.concatenate([x, ds3])

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    up3 = layers.LeakyReLU(alpha=0.3)(x)

    x = layers.Conv2DTranspose(32, (3, 3), (1, 2), padding='same')(up3)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.concatenate([x, ds2])

    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    up2 = layers.LeakyReLU(alpha=0.3)(x)

    x = layers.Conv2DTranspose(16, (3, 3), (1, 2), padding='same')(up2)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.concatenate([x, ds1])

    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    up1 = layers.LeakyReLU(alpha=0.3)(x)

    return up1

# Define the generator model
def define_generator(input_shape):
    inputs = Input(shape=input_shape)
    encoder_model = encoder(input_shape)
    ds4, ds3, ds2, ds1 = encoder_model(inputs)
    bottleneck_output = bottleneck(ds4)
    decoder_output = decoder(bottleneck_output, ds4, ds3, ds2, ds1)
    
    x = layers.Conv2DTranspose(1, (3, 3), (1, 2), padding='same')(decoder_output)
    outputs = layers.LeakyReLU(alpha=0.3)(x)
    
    generator = Model(inputs, outputs, name="Generator")
    return generator

# Define the custom GAN model
class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, theta1, theta2):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.theta1 = theta1
        self.theta2 = theta2
        self.cross_entropy = BinaryCrossentropy()

    def compile(self, d_optimizer, g_optimizer, **kwargs):
        super(GAN, self).compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def train_step(self, data):
        X_train, Y_train = data

        # Train the discriminator
        with tf.GradientTape() as tape:
            fake_output = self.generator(X_train, training=True)
            real_output = self.discriminator([X_train, Y_train], training=True)
            fake_pred = self.discriminator([X_train, fake_output], training=True)
            d_loss = (self.cross_entropy(tf.ones_like(real_output), real_output) +
                      self.cross_entropy(tf.zeros_like(fake_pred), fake_pred)) / 2

        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as tape:
            fake_output = self.generator(X_train, training=True)
            fake_pred = self.discriminator([X_train, fake_output], training=True)
            g_loss = (self.cross_entropy(tf.ones_like(fake_pred), fake_pred) +
                      self.theta1 * tf.reduce_mean(tf.square(Y_train - fake_output)))

        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        total_gan_loss = d_loss + self.theta2 * g_loss

        return {"d_loss": d_loss, "g_loss": g_loss, "total_gan_loss": total_gan_loss}




# Hyperparameters
theta1 = 0.1
theta2 = 0.1
input_shape = (3, 110240, 1)  # Adjust as necessary

# Create the models
discriminator = define_discriminator(input_shape)
generator = define_generator(input_shape)

# Compile and train the GAN
gan = GAN(discriminator, generator, theta1, theta2)
gan.compile(d_optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            g_optimizer=Adam(learning_rate=0.0002, beta_1=0.5))


path1 = "/home/anchal/Desktop/rajesh/Clean/Dataset/MUSDBHQ/CM/XtrainCM3.npy"
path2 = "/home/anchal/Desktop/rajesh/Clean/Dataset/MUSDBHQ/CM/YtrainCM3.npy"

X_train =  np.load(path1)
Y_train =  np.load(path2) # np.random.random((10, 3, 110240, 1))

#print('\n\n\n\n', X_train.shape, Y_train.shape)
epochs = 10
batch_size = 4
gan.fit(np.expand_dims(X_train, axis=-1), np.expand_dims(Y_train, axis=-1), epochs=epochs, batch_size=batch_size)
generator.save('generator_model_final.h5')
