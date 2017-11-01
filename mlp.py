#!/bin/python2
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import os
from extract import *

# Hyperparametes
test_split = 0.3
num_epochs = 10
batch_size = 32

# Model definition
model = Sequential()
model.add(Dense(512, input_dim=3197, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Extract, shuffle and split data
train_data, train_labels, test_data, test_labels = split_data(
                                                       test_split=test_split)

# Train and evaluate
model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size)
print model.evaluate(test_data, test_labels, batch_size=256)

