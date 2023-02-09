import numpy as np
from Models import model_with_attention, model_without_attention
from utils import *
from keras.callbacks import ModelCheckpoint
from ReadData import ReadData
import random
import tensorflow as tf
import os

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["KERAS_BACKEND"] = "tensorflow"

random.seed(10)
# configs
datatype = "freeway"
# urban datatype does not exist in the data directory. If you would like to test it, download the data from the original repository and put it in data/urban directory
epoch = 15
seq_len = 15
pre_len = 12
pre_sens_num = 1
batch_size = 128
model_type = "with_attention"
# model_type = "without_attention"
demo = True

Data = ReadData(
    datatype=datatype,
    seq_len=seq_len,
    pre_len=pre_len,
    pre_sens_num=pre_sens_num,
    demo=demo,
)
if model_type == "with_attention":
    model = model_with_attention()
    model.summary()
    print("Model with attention is called.")
elif model_type == "without_attention":
    model = model_without_attention()
    model.summary()
    print("Model without attention is called.")
else:
    print("model type is not valid.")

model.compile(optimizer="adam", loss=my_loss)

# # train_save model
# filepath = "model/model_{epoch:04d}-{val_loss:.4f}.h5"
# checkpoint = ModelCheckpoint(
#     filepath, monitor="val_loss", verbose=1, save_best_only=True, mode="min", period=5
# )
# callbacks_list = [checkpoint]

print("Training:")
model.fit(
    x=[Data.train_data, Data.train_w, Data.train_d],
    y=Data.label,
    batch_size=batch_size,
    epochs=epoch,
    validation_split=0.15,
    # callbacks=callbacks_list,
)

predicted = predict_point_by_point(model, [Data.test_data, Data.test_w, Data.test_d])
p_real = []
l_real = []
for i in range(Data.test_data.shape[0]):
    p_real.append(predicted[i] * Data.test_med + Data.test_min)
    l_real.append(Data.test_l[i] * Data.test_med + Data.test_min)
p_real = np.array(p_real)
l_real = np.array(l_real)

print("MAE:", MAE(p_real, l_real))
print("MAPE:", MAPE(p_real, l_real))
print("RMSE:", RMSE(p_real, l_real))
print("MSE:", RMSE(p_real, l_real) ** 2)
