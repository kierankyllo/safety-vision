# this script was executed in the google colab training environment with access to training data preprocessed and stored on a mapped google drive

!pip install -q tflite-model-maker
!pip install -q pycocotools
!pip install numpy --upgrade

import numpy as np
import os

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# define source folders for notes and images
train_images = '/content/drive/MyDrive/CS490/data/train/images'
test_images = '/content/drive/MyDrive/CS490/data/test/images'
val_images = '/content/drive/MyDrive/CS490/data/val/images'

train_notes = '/content/drive/MyDrive/CS490/data/train/annotations'
test_notes = '/content/drive/MyDrive/CS490/data/test/annotations'
val_notes = '/content/drive/MyDrive/CS490/data/val/annotations'

# define the labelmap
labelmap = {1:'helmet', 2:'person', 3:'head'}

# build the dataloader objects
train_data = object_detector.DataLoader.from_pascal_voc(train_images, train_notes, labelmap)
test_data = object_detector.DataLoader.from_pascal_voc(test_images, test_notes, labelmap)
val_data = object_detector.DataLoader.from_pascal_voc(val_images, val_notes, labelmap)

# train the object detector model 
model = object_detector.create(train_data, model_spec=spec, epochs=100, batch_size=8, train_whole_model=False, do_train=True, validation_data=val_data)

# evaluate the model performance
model.evaluate(test_data)

# setup model export
export_path = '/content/drive/MyDrive/CS490/heavy_model'

# export the three types of tflite models
model.export(export_dir=export_path, export_format=[ExportFormat.TFLITE, ExportFormat.SAVED_MODEL, ExportFormat.LABEL])

# evaluate the quantized model
model.evaluate_tflite(export_path + '/model.tflite', test_data)

