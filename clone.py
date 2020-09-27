# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 23:05:24 2020

@author: jnnascimento
"""

import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from sklearn.model_selection import train_test_split
import sklearn
import math
import random
import matplotlib.pyplot as plt

# Read all lines from the driving csv file
samples  = []
with open('training_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Slip the train and validation samples using the rate of 80% / 20%
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Since the whole samples images size are pretty huge ( more than 10GB!!)
# lets use a generator with batches to progressively load data to the training
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                # Lets use both the right and left images for training the model
                # and consider a correction factor for them
                correction = 0.2
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                for i in range(3):
                    source_path = batch_sample[i]
                    # Change the "\\" if you are using linux
                    filename = source_path.split('\\')[-1]
                    current_path = 'training_data/IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                measurements.append(steering_center)
                measurements.append(steering_left)
                measurements.append(steering_right)
            
            # In order to augment the input samples, we are going to duplicate
            # all the input images and flip them
            augmented_images, augmented_measurements = [], []
            for image, measurements in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurements)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurements*-1.0)
                
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            
            yield sklearn.utils.shuffle(X_train, y_train)
            
# Set our batch size
batch_size=64

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Model architecture
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24, (5, 5), strides = (2,2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides = (2,2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides = (2,2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator,
            steps_per_epoch=math.ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=math.ceil(len(validation_samples)/batch_size),
            epochs=20, verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
