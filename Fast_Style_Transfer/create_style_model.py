import tensorflow as tf
from train_premade_style import *
from preprocess_util import *


import tensorflow.python.client as device_lib
print(tf.__version__)
print(device_lib.device_lib.list_local_devices())
model_name = 'starrynight'

style_model = train_style_model('C:\Python Files\pre_made_style_transfer\style_images\starry_night.jpg', 'D:\coco_data/train/data/', use_gpu=True)
style_model.save(f'{model_name}.h5')