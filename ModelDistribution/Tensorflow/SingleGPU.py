import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pyprojroot import here

batch_size = 64
epochs = 1000
input_shape = (120, 5, 5, 13)
horizon = 6
lr = 0.001
early_stop_paitience = 12
reduce_lr_paitience = 3
reduce_lr_factor = 0.6

x_train_path = here("data/3d_regression/x_train.npy")
y_train_path = here("data/3d_regression/y_train.npy")
x_valid_path = here("data/3d_regression/x_valid.npy")
y_valid_path = here("data/3d_regression/y_valid.npy")

# Design a Convolutional/LSTM benchmark. 
# (we will play around with adding and removing the lstms to see the affect on model distribution on the GPU)
def benchmark(input_shape, horizon):
    input_shape = input_shape
    input = tf.keras.layers.Input(shape=input_shape)

    conv = tf.keras.layers.Conv3D(filters=13, kernel_size=(1, 3, 3), activation='relu', padding='same')(input)
    conv = tf.keras.layers.Conv3D(filters=13, kernel_size=(1, 3, 3), activation='relu', padding='same')(conv)
    x = tf.keras.layers.Reshape((120, 25, 13))(conv)

    conv = tf.keras.layers.Conv2D(13, kernel_size=(1,3), padding="same", activation="relu")(x)
    conv = tf.keras.layers.Conv2D(13, kernel_size=(1,3), padding="same", activation="relu")(conv)
    x = tf.keras.layers.Reshape((120, 325))(conv)

    lstm = tf.keras.layers.LSTM(13, return_sequences=True)(x)
    lstm = tf.keras.layers.LSTM(13, return_sequences=True)(lstm)
    flatten = tf.keras.layers.Flatten()(lstm)
    
    y = tf.keras.layers.Dense(25, activation='relu')(flatten)
    y = tf.keras.layers.Dropout(0.1)(y)
    out = tf.keras.layers.Dense(horizon, activation='linear')(y)
    model = tf.keras.models.Model(inputs=input, outputs=out)
    model.summary()
    return model



steps_per_epoch = len(np.load(x_train_path))//batch_size
x_train_shape = np.load(x_train_path).shape
print("steps_per_epoch:", steps_per_epoch)

val_steps_per_epoch = len(np.load(x_valid_path))//batch_size
x_valid_shape = np.load(x_valid_path).shape
print("val_steps_per_epoch:", val_steps_per_epoch)

# Create a function to load the data in batches
def load_data_in_batches(x_path, y_path, data_shape, batch_size, num_epochs=epochs):
    # Compute number of batches
    num_batches = data_shape[0] // batch_size
    for epoch in range(num_epochs):
        # Loop over batches and yield data
        for i in range(num_batches):
            # Load batch from numpy files
            x_batch = np.load(x_path, mmap_mode='r')[i*batch_size:(i+1)*batch_size]
            y_batch = np.load(y_path, mmap_mode='r')[i*batch_size:(i+1)*batch_size]
            y_batch = y_batch.reshape(y_batch.shape[0], horizon)
            # Convert to TensorFlow tensors
            x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
            y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
            # Yield batch
            yield x_batch, y_batch


options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

# Create the TensorFlow datasets
train_generator = tf.data.Dataset.from_generator(
    lambda: load_data_in_batches(x_train_path, y_train_path, x_train_shape, batch_size),
    output_signature=(
        tf.TensorSpec(shape=(batch_size, 120, 5, 5, 13), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size, horizon), dtype=tf.float32)
    )
).with_options(options).repeat()

valid_generator = tf.data.Dataset.from_generator(
    lambda: load_data_in_batches(x_valid_path, y_valid_path, x_valid_shape, batch_size),
    output_signature=(
        tf.TensorSpec(shape=(batch_size, 120, 5, 5, 13), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size, horizon), dtype=tf.float32)
    )
).with_options(options).repeat()

# strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
strategy = tf.distribute.MirroredStrategy()

# utilize the strategy and wrap model compile, and training phase into it
with strategy.scope():
    model = benchmark(input_shape, horizon)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
        )
    
    metrics = [tf.keras.metrics.MeanAbsoluteError(name="mae")]
    model.compile(loss="mse", optimizer=optimizer, metrics=metrics)

    early_stop = EarlyStopping(monitor='val_loss', patience=early_stop_paitience, restore_best_weights=True, verbose=1, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor, patience=reduce_lr_paitience, min_lr=1e-7)
    # train using the datagenerators
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_generator,
        validation_steps=val_steps_per_epoch,
        callbacks=[early_stop, reduce_lr],
        verbose=1
        )

    # Evaluate the model on test
    x_test = np.load("data/3d_regression/x_test.npy")
    y_test = np.load("data/3d_regression/y_test.npy")
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("test loss:", results)