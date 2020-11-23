from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

test_dir = 'test'

test_interesting_dir = os.path.join(test_dir, 'interesting')
test_uninteresting_dir = os.path.join(test_dir, 'uninteresting')

num_interesting_val = len(os.listdir(test_interesting_dir))
num_uninteresting_val = len(os.listdir(test_uninteresting_dir))

total_val = num_interesting_val + num_uninteresting_val

IMG_WIDTH = 160
IMG_HEIGHT = 160

test_image_generator = ImageDataGenerator(rescale=1./255)

print('Testing data:')
test_data_gen = test_image_generator.flow_from_directory(directory=test_dir,
                                                         target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                         batch_size=5,
                                                         class_mode='binary')

model = tf.keras.models.load_model('model_weights/vgg19.h5')
model.summary()

print("Start time:", datetime.now())
print()

results = model.evaluate(test_data_gen)

print()
print("End time:", datetime.now())
print()
print('loss:', results[0])
print('binary_accuracy:', results[1])
print('precision:', results[2])
print('recall', results[3])