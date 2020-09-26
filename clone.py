# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 23:05:24 2020

@author: jnnascimento
"""

import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('training_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    steering_center = float(line[3])
    correction = 0.2
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    for i in range(3):        
        source_path = line[i]
        # Change the "\\" if you are using linux
        filename = source_path.split('\\')[-1]
        current_path = 'training_data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)
    
augmented_images, augmented_measurements = [], []
for image, measurements in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurements)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurements*-1.0)

print(len(augmented_images))

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# model = Sequential()
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
# model.add(Flatten())
# model.add(Dense(1))

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(6, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(16, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')

