from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

datadir = 'data'
train_dir = os.path.join(datadir, 'train_data')
test_dir = os.path.join(datadir, 'test_data')

train_interesting_dir = os.path.join(train_dir, 'interesting')
train_uninteresting_dir = os.path.join(train_dir, 'uninteresting')
test_interesting_dir = os.path.join(test_dir, 'interesting')
test_uninteresting_dir = os.path.join(test_dir, 'uninteresting')

num_interesting_tr = len(os.listdir(train_interesting_dir))
num_uninteresting_tr = len(os.listdir(train_uninteresting_dir))
num_interesting_val = len(os.listdir(test_interesting_dir))
num_uninteresting_val = len(os.listdir(test_uninteresting_dir))

total_train = num_interesting_tr + num_uninteresting_tr
total_val = num_interesting_val + num_uninteresting_val

batch_size = 32
epochs = 8
IMG_WIDTH = 160
IMG_HEIGHT = 160

train_image_generator = ImageDataGenerator(rescale=1./255,
                                           rotation_range=45,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           zoom_range=0.1,
                                           width_shift_range=.1,
                                           height_shift_range=.1
                                        )
test_image_generator = ImageDataGenerator(rescale=1./255)

print('Training data:')
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                           class_mode='binary')

print('Testing data:')
test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                              class_mode='binary')

base_model = InceptionV3(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
                          include_top=False,
                          weights='imagenet')
base_model.trainable = False

model = Sequential([
  base_model,
  GlobalAveragePooling2D(),
  Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[BinaryAccuracy(), Precision(), Recall()])

model.summary()

print("Start time:", datetime.now())
print()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=test_data_gen,
    validation_steps=total_val // batch_size
)

print()
print("End time:", datetime.now())

model.save('model_weights/inception.h5')

acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 4, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Test Accuracy')
plt.legend()
plt.title('Training and Test Accuracy')

plt.subplot(1, 4, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Test Loss')
plt.legend()
plt.title('Training and Test Loss')

plt.subplot(1, 4, 3)
plt.plot(epochs_range, precision, label='Training Precision')
plt.plot(epochs_range, val_precision, label='Test Precision')
plt.legend()
plt.title('Training and Test Precision')

plt.subplot(1, 4, 4)
plt.plot(epochs_range, recall, label='Training Recall')
plt.plot(epochs_range, val_recall, label='Test Recall')
plt.legend()
plt.title('Training and Test Recall')
plt.show()