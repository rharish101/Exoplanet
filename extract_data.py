#!/bin/python2
import numpy as np

csv = open('ExoTrain.csv', 'r')
csv_data = csv.read()
csv.close()

train_data = np.array([vec.split(',')[1:] for vec in csv_data.split(
                       '\r\n')[1:-1]]).astype(np.float32)
train_labels = np.array([int(vec.split(',')[0]) - 1 for vec in csv_data.split(
                         '\r\n')[1:-1]])
np.save(open('data.npy', 'w'), train_data)
np.save(open('labels.npy', 'w'), train_labels)

