#!/bin/python2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
import numpy as np
import os

# Hyperparametes
test_split = 0.3
num_epochs = 20
batch_size = 32

# Model definition
model = Sequential()
model.add(LSTM(256, kernel_initializer='glorot_normal',
               bias_initializer='constant', input_shape=(3197, 1)))
model.add(Dense(1, kernel_initializer='glorot_normal',
               bias_initializer='constant', activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',
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

# Normalize data
data = (data - np.mean(data, -1, keepdims=True)) / np.std(data, -1,
                                                          keepdims=True)

# Shuffle data
combined = np.column_stack((data, labels))
np.random.shuffle(combined)
data = combined.T[:-1].T
labels = combined.T[-1].T
data = np.reshape(data, (-1, 3197, 1))

# Split dataset
train_data = data[:-int(len(data) * test_split)]
train_labels = labels[:-int(len(labels) * test_split)]
test_data = data[-int(len(data) * test_split):]
test_labels = labels[-int(len(labels) * test_split):]

# Train and evaluate
early_stop = EarlyStopping(monitor='loss', min_delta=1e-4, patience=3)
model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size,
          callbacks=[early_stop])
print "Calculating test accuracy..."
results = model.evaluate(test_data, test_labels, batch_size=32)
print "Test accuracy:%6.2f%%" % (results[1] * 100)
