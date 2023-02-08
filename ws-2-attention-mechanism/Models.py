import keras
from utils import *
from keras.layers import Bidirectional
from keras.layers.core import Dense, Flatten
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.layers.convolutional import Conv1D
from AttentionLayer import Attention
from AttentionWithContextLayer import AttentionWithContext
from keras.layers.wrappers import TimeDistributed


def model_with_attention():
    # conv-lstm: injecting train_data
    main_input = Input((15, 7, 1), name="main_input")
    con1 = TimeDistributed(
        Conv1D(filters=15, kernel_size=3, padding="same", activation="relu", strides=1)
    )(main_input)
    con2 = TimeDistributed(
        Conv1D(filters=15, kernel_size=3, padding="same", activation="relu", strides=1)
    )(con1)
    # con3 = TimeDistributed(AveragePooling1D(pool_size=2))(con2)
    con_fl = TimeDistributed(Flatten())(con2)
    con_out = Dense(15)(con_fl)

    lstm_out1 = LSTM(15, return_sequences=True)(con_out)
    # lstm_attention = AttentionWithContext()(lstm_out1)
    lstm_out2 = LSTM(15, return_sequences=False)(lstm_out1)
    lstm_out3 = Attention()([lstm_out2, con_out])

    # Bilstm
    # injecting train_w
    auxiliary_input_w = Input((15, 1), name="auxiliary_input_w")
    lstm_outw1 = Bidirectional(LSTM(15, return_sequences=True))(auxiliary_input_w)
    lstm_outw2 = Bidirectional(LSTM(15, return_sequences=False))(lstm_outw1)

    # injecting train_d
    auxiliary_input_d = Input((15, 1), name="auxiliary_input_d")
    lstm_outd1 = Bidirectional(LSTM(15, return_sequences=True))(auxiliary_input_d)
    lstm_outd2 = Bidirectional(LSTM(15, return_sequences=False))(lstm_outd1)

    x = keras.layers.concatenate([lstm_out3, lstm_outw2, lstm_outd2])
    x = Dense(20, activation="relu")(x)
    x = Dense(10, activation="relu")(x)
    main_output = Dense(
        1,
        activation="relu",
        kernel_regularizer=keras.regularizers.l1_l2(0.1, 0.1),
        name="main_output",
    )(x)
    model = Model(
        inputs=[main_input, auxiliary_input_w, auxiliary_input_d], outputs=main_output
        )
    return model

def model_without_attention():
    # conv-lstm: injecting train_data
    main_input = Input((15, 7, 1), name="main_input")
    con1 = TimeDistributed(
        Conv1D(filters=15, kernel_size=3, padding="same", activation="relu", strides=1)
    )(main_input)
    con2 = TimeDistributed(
        Conv1D(filters=15, kernel_size=3, padding="same", activation="relu", strides=1)
    )(con1)
    con_fl = TimeDistributed(Flatten())(con2)
    con_out = Dense(15)(con_fl)

    lstm_out1 = LSTM(15, return_sequences=True)(con_out)
    lstm_out2 = LSTM(15, return_sequences=False)(lstm_out1)

    # Bilstm
    # injecting train_w
    auxiliary_input_w = Input((15, 1), name="auxiliary_input_w")
    lstm_outw1 = Bidirectional(LSTM(15, return_sequences=True))(auxiliary_input_w)
    lstm_outw2 = Bidirectional(LSTM(15, return_sequences=False))(lstm_outw1)

    # injecting train_d
    auxiliary_input_d = Input((15, 1), name="auxiliary_input_d")
    lstm_outd1 = Bidirectional(LSTM(15, return_sequences=True))(auxiliary_input_d)
    lstm_outd2 = Bidirectional(LSTM(15, return_sequences=False))(lstm_outd1)

    x = keras.layers.concatenate([lstm_out2, lstm_outw2, lstm_outd2])
    x = Dense(20, activation="relu")(x)
    x = Dense(10, activation="relu")(x)
    main_output = Dense(
        1,
        activation="relu",
        kernel_regularizer=keras.regularizers.l1_l2(0.1, 0.1),
        name="main_output",
    )(x)
    model = Model(
        inputs=[main_input, auxiliary_input_w, auxiliary_input_d], outputs=main_output
    )
    return model