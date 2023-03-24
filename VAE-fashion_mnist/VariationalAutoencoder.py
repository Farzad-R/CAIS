"""
This code implements a Variational Autoencoder (VAE) to generate fashion MNIST images.

Code breakdown:
- It loads the fashion MNIST dataset, 
- normalizes and flattens the images
- defines the encoder and decoder models
- defines the VAE model
- trains it. 
- Finally, it generates some fake images using the trained decoder. 

The VAE model is trained to minimize the sum of the reconstruction loss and the Kullback-Leibler (KL) divergence 
between the learned distribution and the prior distribution of the latent variables. The decoder generates the fake
images by sampling from a normal distribution and decoding the sampled latent variables.

=================================================
Overview of Variational Autoencoder (VAE):

It is a type of generative neural network model that learns to generate new data similar to the training data. 
It consists of an encoder and a decoder network that are trained together using a special loss function that encourages
the learned representations to follow a particular distribution, typically a normal or Gaussian distribution. This 
encourages the model to learn a compressed and structured representation of the input data that can be used to generate
new samples by sampling from the learned distribution. VAEs are used in a variety of applications, including image and 
video generation, anomaly detection, and data compression.
=================================================
"""

import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping


# Load the Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the images
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Flatten the images
original_dim = 784
x_train = x_train.reshape((len(x_train), original_dim))
x_test = x_test.reshape((len(x_test), original_dim))

input_shape = (original_dim,)

# Dimensions of the latent space
latent_dim = 2

# The size of the intermediate layer
intermediate_dim = 256
batch_size = 128
epochs = 10000

# Define the sampling function for the VAE
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Define the encoder model
inputs = Input(shape=input_shape)
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)
z = Lambda(sampling)([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
# plot_model(encoder, to_file='encoder.png', show_shapes=True)

# Define the decoder model
latent_inputs = Input(shape=(latent_dim,))
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
# plot_model(decoder, to_file='decoder.png', show_shapes=True)

# Define the VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

# Define the loss function for the VAE
reconstruction_loss = mse(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Define the EarlyStopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5)
# Train the VAE model
history = vae.fit(
    x_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, None),
    callbacks=[early_stop]
    )

# Generate some fake images using the decoder
n = 10
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = np.linspace(-4, 4, n)
grid_y = np.linspace(-4, 4, n)[::-1]
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit
        # Plot the generated images
        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.axis('off')
        plt.show()