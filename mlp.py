#!/bin/python2
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import os

# Hyperparametes
test_split = 0.3
num_epochs = 10
batch_size = 32

# Model definition
model = Sequential()
model.add(Dense(512, input_dim=3197, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='sgd', loss='binary_crossentropy',
              metrics=['accuracy'])

# Extract data
if 'data.npy' not in os.listdir('.') or 'labels.npy' not in os.listdir('.'):
    if 'ExoTrain.csv' in os.listdir('.'):
        csv = open('ExoTrain.csv', 'r')
    else:
        print "Dataset missing"
        exit()
    csv_data = csv.read()
    csv.close()
    train_data = np.array([vec.split(',')[1:] for vec in csv_data.split(
                        '\r\n')[1:-1]]).astype(np.float32)
    train_labels = np.array([int(vec.split(',')[0]) - 1 for vec in\
                             csv_data.split('\r\n')[1:-1]])
    np.save(open('data.npy', 'w'), train_data)
    np.save(open('labels.npy', 'w'), train_labels)
else:
    data = np.load(open('data.npy', 'r'))
    labels = np.load(open('labels.npy', 'r'))

# Shuffle data
combined = np.dstack((data, labels))[0]
np.random.shuffle(combined)
data, labels = combined.T

# Split dataset
train_data = data[:-int(len(data) * test_split)]
train_labels = labels[:-int(len(labels) * test_split)]
test_data = data[-int(len(data) * test_split):]
test_labels = labels[-int(len(labels) * test_split):]

# Train and evaluate
model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size)
print model.evaluate(test_data, test_labels, batch_size=256)

