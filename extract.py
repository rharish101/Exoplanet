#!/bin/python2
import numpy as np
import os

def extract_data():
    if 'data.npy' not in os.listdir('.') or 'labels.npy' not in os.listdir('.'):
        if 'ExoTrain.csv' in os.listdir('.'):
            csv = open('ExoTrain.csv', 'r')
        else:
            print "Dataset missing"
            exit()
        csv_data = csv.read()
        csv.close()
        data = np.array([vec.split(',')[1:] for vec in csv_data.split(
                            '\r\n')[1:-1]]).astype(np.float32)
        labels = np.array([int(vec.split(',')[0]) - 1 for vec in\
                                csv_data.split('\r\n')[1:-1]])
        np.save(open('data.npy', 'w'), data)
        np.save(open('labels.npy', 'w'), labels)
    else:
        data = np.load(open('data.npy', 'r'))
        labels = np.load(open('labels.npy', 'r'))
    return data, labels

def shuffle_data(data=None, labels=None):
    if data is None or labels is None:
        data, labels = extract_data()
    combined = np.column_stack((data, labels))
    np.random.shuffle(combined)
    data = combined.T[:-1].T
    labels = combined.T[-1].T
    return data, labels

def split_data(data=None, labels=None, shuffle=True):
    if data is None or labels is None:
        data, labels = extract_data()
    if shuffle:
        data, labels = shuffle_data(data, labels)
    train_data = data[:-int(len(data) * test_split)]
    train_labels = labels[:-int(len(labels) * test_split)]
    test_data = data[-int(len(data) * test_split):]
    test_labels = labels[-int(len(labels) * test_split):]
    return train_data, train_labels, test_data, test_labels

