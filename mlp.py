#!/bin/python2
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

model = Sequential()
model.add(Dense(512, input_dim=3197, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='sgd', loss='binary_crossentropy',
              metrics=['accuracy'])

data = np.load(open('data.npy', 'r'))
labels = np.load(open('labels.npy', 'r'))
combined = np.dstack((data, labels))[0]
np.random.shuffle(combined)
data, labels = combined.T

test_split = 0.3
train_data = data[:-int(len(data) * test_split)]
train_labels = labels[:-int(len(labels) * test_split)]
test_data = data[-int(len(data) * test_split):]
test_labels = labels[-int(len(labels) * test_split):]

num_epochs = 10
batch_size = 32
model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size)

print model.evaluate(test_data, test_labels, batch_size=256)

