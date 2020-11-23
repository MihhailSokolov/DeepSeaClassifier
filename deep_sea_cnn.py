from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
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

print('total training "interesting" images:', num_interesting_tr)
print('total training "uninteresting" images:', num_uninteresting_tr)

print('total test "interesting" images:', num_interesting_val)
print('total test "uninteresting" images:', num_uninteresting_val)
print("--")
print("Total training images:", total_train)
print("Total test images:", total_val)
print()
batch_size = 64
epochs = 30
IMG_WIDTH = 512
IMG_HEIGHT = 256

train_image_generator = ImageDataGenerator(rescale=1./255,
                                           rotation_range=45,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           zoom_range=0.1,
                                           width_shift_range=.1,
                                           height_shift_range=.1
                                        )
test_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                           class_mode='binary')

test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                              class_mode='binary')

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    # MaxPooling2D(),
    # Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    # Dropout(0.2),
    Flatten(),
    # Dense(512, activation='relu'),
    Dense(64, activation='relu'),
    # Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[
                Accuracy(),
                Precision(),
                Recall()
              ])

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

model.save('model_weights/model.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

precision = history.history['precision']
val_precision = history.history['val_precision']

recall = history.history['recall']
val_recall = history.history['val_recall']

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