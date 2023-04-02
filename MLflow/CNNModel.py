import tensorflow as tf



def build_cnnmodel():
# Designing the model architecture
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(28,28,1)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size = (3,3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.BatchNormalization())    

    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten(input_shape=[28,28]))
    model.add(tf.keras.layers.Dense(300, activation="relu"))
    model.add(tf.keras.layers.Dense(100, activation="relu"))
    # last layer
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.summary()
    return model