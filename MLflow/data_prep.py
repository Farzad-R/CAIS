
from keras.utils.np_utils import to_categorical
import tensorflow as tf

def prepare_mnist_data():
    # Loading the data
    mnist = tf.keras.datasets.mnist
    (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()
    x_train_full = x_train_full.reshape(x_train_full.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    print("Example before converting to categorical:", y_train_full[0])
    y_train_full = to_categorical(y_train_full)
    y_test = to_categorical(y_test)
    print("Example after converting to categorical:", y_train_full[0])

    # normalizing the training data and spliting the data into trian and validation sets
    x_test = x_test / 255.0
    x_valid = x_train_full[:5000] / 255.0
    x_train = x_train_full[5000:] / 255.0
    y_valid = y_train_full[:5000]
    y_train = y_train_full[5000:]

    print("train set: x_train:", x_train.shape, "y_train:", y_train.shape)
    print("valid set: x_valid:", x_valid.shape, "y_valid:", y_valid.shape)
    print("test set: xtest:", x_test.shape, "y_test:", y_test.shape)
    return x_train, y_train, x_valid, y_valid, x_test, y_test