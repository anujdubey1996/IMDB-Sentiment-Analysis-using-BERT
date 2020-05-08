import ModelGenerator as mg
import datetime
import tensorflow as tf
from tensorflow import keras

CUDA_VISIBLE_DEVICES = 0,1

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()

log_dir = "../log/intent_detection/" + datetime.datetime.now().strftime("%m%d%H%M%S/")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

class Trainer:
  def __init__(self, model, preprocessed_data):
    self.history = model.fit(
    x=preprocessed_data.train_x, 
    y=preprocessed_data.train_y,
    validation_split=0.1,
    batch_size=256,
    shuffle=True,
    epochs=5,
    callbacks=[keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = 100000000)])
