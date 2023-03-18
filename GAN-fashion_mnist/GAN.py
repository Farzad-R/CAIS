"""
This code trains a Generative Adversarial Network (GAN) using the Fashion-MNIST dataset to generate 
fake images. The generator model generates fake images from random noise while the discriminator 
model learns to distinguish between real and fake images. The GAN model combines the generator 
and discriminator models. The GAN is trained to minimize the loss of the discriminator in 
distinguishing real and fake images and maximize the loss of the generator in fooling the 
discriminator. The code outputs the progress after each epoch and generates images for visualization.


Generative Adversarial Network (GAN):
It is a type of generative model that consists of two neural networks working in tandem:
a generator and a discriminator.

The generator takes a random noise vector as input and generates synthetic data (e.g., images) that 
resemble the real data. The discriminator is a binary classifier that distinguishes between real and 
synthetic data.

The training process of GAN involves a competition between the generator and discriminator. The 
generator tries to produce synthetic data that can fool the discriminator, while the discriminator 
tries to distinguish the synthetic data from real data. The generator is trained to minimize the 
probability that the discriminator correctly identifies synthetic data, while the discriminator is 
trained to maximize its ability to correctly identify real and synthetic data.

As the generator and discriminator are trained together, the generator becomes better at generating 
synthetic data that can fool the discriminator, and the discriminator becomes better at distinguishing 
between real and synthetic data. This process continues until the generator produces synthetic data 
that is indistinguishable from real data, and the discriminator can no longer differentiate between 
the two.

The architecture of a typical GAN consists of a generator network and a discriminator network. 
The generator network typically consists of one or more fully connected layers followed by one 
or more convolutional transpose layers, while the discriminator network consists of one or more 
convolutional layers followed by one or more fully connected layers. Both networks may also use 
batch normalization, dropout, and activation functions such as ReLU or LeakyReLU.
"""

# Import the librarties
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2DTranspose, Conv2D, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential

print("tensorflow version:", tf.__version__)
print("numpy version:", np.__version__)

# Load the Fashion-MNIST dataset
(train_images, train_labels), (_, _) = fashion_mnist.load_data()


# Normalize the images
train_images = train_images / 255.0

# Design the generator model
generator = Sequential([
    Dense(7*7*256, input_shape=(100,)),
    Reshape((7, 7, 256)),
    Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
    LeakyReLU(alpha=0.2),
    Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
    LeakyReLU(alpha=0.2),
    Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
])

# Define the discriminator model
discriminator = Sequential([
    Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
    LeakyReLU(alpha=0.2),
    Dropout(0.3),
    Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
    LeakyReLU(alpha=0.2),
    Dropout(0.3),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# Compile the discriminator model
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])

# Design the GAN model
gan = Sequential([
    generator,
    discriminator
])

# Compile the GAN model
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))


# Train the GAN model
epochs = 100
batch_size = 128
steps_per_epoch = int(train_images.shape[0] / batch_size)
num_images = 25

for epoch in range(epochs):
    for step in range(steps_per_epoch):
        # Train the discriminator on real and fake images
        real_images = train_images[np.random.randint(0, train_images.shape[0], size=batch_size)]
        real_images = np.expand_dims(real_images, axis=-1)
        fake_images = generator.predict(np.random.normal(0, 1, [batch_size, 100]))
        x = np.concatenate([real_images, fake_images])
        y = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
        d_loss = discriminator.train_on_batch(x, y)

        # Train the generator to fool the discriminator
        noise = np.random.normal(0, 1, [batch_size, 100])
        y = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, y)

    # Output the progress after each epoch
    print(f"Epoch {epoch+1}/{epochs}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

    # Output generated images for visualization
    if (epoch+1) % 10 == 0:
        generated_images = generator.predict(np.random.normal(0, 1, [num_images, 100]))
        generated_images = generated_images.reshape(num_images, 28, 28)
        plt.figure(figsize=(10,10))
        for i in range(num_images):
            plt.subplot(5, 5, i+1)
            plt.imshow(generated_images[i], cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()