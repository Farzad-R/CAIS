import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = (x_train.astype('float32') - 127.5) / 127.5
x_test = (x_test.astype('float32') - 127.5) / 127.5

# Add a channel dimension to the images
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

# Set the dimensions of the input noise
latent_dim = 100

# Define the generator model
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(7*7*256, input_dim=latent_dim))
    model.add(Reshape((7, 7, 256)))
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
    model.add(Activation('tanh'))
    return model

# Define the discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(28, 28, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Build the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Define the hyperparameters
latent_dim = 100 # noise dimension
epochs = 50
batch_size = 128
sample_interval = 1000

# Build the generator and discriminator
generator = build_generator(latent_dim)
discriminator = build_discriminator()

# Build the GAN model
gan = build_gan(generator, discriminator)

# Compile the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Compile the GAN model
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Function to sample and save generated images
def sample_images(generator, epoch, latent_dim):
    # Generate a batch of noise vectors
    noise = np.random.normal(0, 1, (25, latent_dim))

    # Generate images from the noise vectors
    gen_imgs = generator.predict(noise)

    # Rescale the images to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Plot the generated images
    fig, axs = plt.subplots(5, 5)
    count = 0
    for i in range(5):
        for j in range(5):
            axs[i,j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
            axs[i,j].axis('off')
            count += 1
    fig.savefig(f"GAN-fashion_mnist/images/{epoch}.png")
    plt.close()


# Define the number of epochs and batch size
epochs = 10000
batch_size = 128

# Define the number of steps per epoch
steps_per_epoch = x_train.shape[0] // batch_size

# Define a list to store the discriminator and generator losses
d_losses = []
g_losses = []
d_loss_epoch = 0
g_loss_epoch = 0
# Train the GAN model
for epoch in range(epochs):
    for batch in tqdm(range(steps_per_epoch)):
        # Sample a batch of noise vectors
        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Generate a batch of fake images
        fake_images = generator.predict(noise)

        # Concatenate the real and fake images
        real_images = x_train[batch * batch_size:(batch + 1) * batch_size]
        combined_images = np.concatenate([fake_images, real_images])

        # Labels for generated and real data
        y_dis = np.zeros(2 * batch_size)
        y_dis[:batch_size] = 1

        # Train the discriminator
        discriminator.trainable = True
        discriminator_loss = discriminator.train_on_batch(combined_images, y_dis)

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        y_gen = np.ones(batch_size)
        discriminator.trainable = False
        gan_loss = gan.train_on_batch(noise, y_gen)

        gan_loss = gan.evaluate(noise, y_gen, verbose=0)
        d_losses.append(discriminator_loss[0])
        g_losses.append(gan_loss)
        d_loss_epoch += discriminator_loss[0]
        g_loss_epoch += gan_loss
    d_loss_epoch = d_loss_epoch/steps_per_epoch
    g_loss_epoch = g_loss_epoch/steps_per_epoch
    # Evaluate the generator on test data
    x_test_noise = np.random.normal(0, 1, (batch_size, latent_dim))
    x_test_generated = generator.predict(x_test_noise)
    x_test_loss = gan.evaluate(x_test_noise, y_gen, verbose=0)
    

    print('Epoch %d: Generator loss=%.4f, Discriminator loss=%.4f, Validation loss=%.4f' % (epoch+1, g_loss_epoch , d_loss_epoch, x_test_loss))
    if epoch // 10 ==0:
        sample_images(generator, epoch, latent_dim)