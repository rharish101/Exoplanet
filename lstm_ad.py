#!/bin/python2
import tensorflow as tf
import numpy as np
import os
import sys
import time
import math
from operator import mul

def weight_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()(shape)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()(shape)
    return tf.Variable(initial)

def LSTM(x, time_steps, hidden_nodes):
    x = tf.reshape(x, [-1, time_steps])
    x = tf.split(x, time_steps, axis=-1)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_nodes)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return outputs[-1]

def LSTMx2(x, time_steps, hidden_nodes_1, hidden_nodes_2):
    x = tf.reshape(x, [-1, time_steps])
    x = tf.split(x, time_steps, axis=-1)
    lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(hidden_nodes_1)
    lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(hidden_nodes_2)
    lstm_stacked = tf.contrib.rnn.MultiRNNCell([lstm_cell1, lstm_cell2])
    outputs, states = tf.contrib.rnn.static_rnn(lstm_stacked, x,
                                                dtype=tf.float32)
    return outputs[-1]

def dense(x, input_shape, num_neurons):
    if len(input_shape) > 2:
        flat = tf.reshape(x, [-1, reduce(mul, input_shape[1:], 1)])
    else:
        flat = x
    W = weight_variable([reduce(mul, input_shape[1:], 1), num_neurons])
    b = bias_variable([num_neurons])
    return tf.matmul(flat, W) + b

#def predict(x):
    #cond = tf.less(x, tf.zeros(tf.shape(x)))
    #out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
    #return out

#def predict_autoenc(x, pred):
    ##threshold = tf.Variable(5.0)
    #threshold = tf.constant(5.0)
    #loss = tf.reduce_mean(tf.square(x - y_pred), axis=-1)
    #cond = tf.less(loss, tf.scalar_mul(threshold, tf.ones(tf.shape(loss))))
    #out = tf.where(cond, tf.zeros(tf.shape(loss)), tf.ones(tf.shape(loss)))
    #return out

def multivariate_normal_diag(x):
    mean, variance = tf.nn.moments(x, axes=0)
    #prob = tf.contrib.distributions.MultivariateNormalDiag(mean,
           #tf.sqrt(variance)).prob(x)
    prob = tf.reduce_prod((1 / (tf.sqrt(2 * math.pi * variance))) * tf.pow(
           math.e, -1 * tf.divide(tf.squared_difference(x,
           mean), 2 * variance)), axis=1)
    #return tf.pow(math.e, -1 * tf.square(prob))
    return 1 / (1e-7 + prob)

def predict_ad(prob):
    threshold = tf.Variable(0.2)
    cond = tf.less(prob, tf.scalar_mul(threshold, tf.ones([tf.shape(x)[0]])))
    out = tf.where(cond, tf.zeros(tf.shape(cond)), tf.ones(tf.shape(cond)))
    return out

def f1_score_func(actual, pred):
    true_positives = tf.to_float(tf.count_nonzero(pred * actual))
    false_positives = tf.to_float(tf.count_nonzero(pred * (actual - 1)))
    false_negatives = tf.to_float(tf.count_nonzero((pred - 1) * actual))
    epsilon = tf.constant(1e-7, dtype=tf.float32)
    precision = tf.divide(true_positives + epsilon,
                          true_positives + false_positives + epsilon)
    recall = tf.divide(true_positives + epsilon,
                       true_positives + false_negatives + epsilon)
    return tf.divide(2 * precision * recall + epsilon,\
                     precision + recall + epsilon)

x = tf.placeholder(tf.float32, [None, 3197])
y_actual = tf.placeholder(tf.float32, [None,])
lstm = LSTM(x, 3197, 128)
#lstm = LSTMx2(x, 3197, 128, 3197)
#y_pred = dense(lstm, [None, 128], 1)
#y_pred = tf.squeeze(y_pred, axis=-1)
#y_pred = dense(lstm, [None, 3197], 3197)
y_pred = multivariate_normal_diag(lstm)

loss = tf.reduce_mean(tf.square(y_actual - y_pred))
#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_actual,
                                                              #logits=y_pred))

#prediction = predict(y_pred)
#prediction = predict_autoenc(x, y_pred)
prediction = predict_ad(y_pred)
f1_score = f1_score_func(y_actual, prediction)
#train_step = tf.train.AdamOptimizer().minimize(loss)
train_step = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9,
                                        use_nesterov=True).minimize(loss)

# Hyperparametes
test_split = 0.3
num_epochs = 10#0
batch_size = 32
display_every = 1
early_stop_threshold = 100
early_stop_patience = 5

# Extract data
if 'data.npy' not in os.listdir('.') or 'labels.npy' not in os.listdir('.'):
    if 'ExoTrain.csv' in os.listdir('.'):
        csv = open('ExoTrain.csv', 'r')
    else:
        print "Dataset missing"
        exit()
    print "Reading dataset..."
    csv_data = csv.read()
    csv.close()
    data = np.array([vec.split(',')[1:] for vec in csv_data.split(
                        '\r\n')[1:-1]]).astype(np.float32)
    labels = np.array([int(vec.split(',')[0]) - 1 for vec in\
                             csv_data.split('\r\n')[1:-1]])
    np.save(open('data.npy', 'w'), data)
    np.save(open('labels.npy', 'w'), labels)
else:
    print "Loading dataset..."
    data = np.load(open('data.npy', 'r'))
    labels = np.load(open('labels.npy', 'r'))
print "Data loaded"

# Normalize data
#data = (data - np.mean(data, -1, keepdims=True)) / np.std(data, -1,
                                                          #keepdims=True)

# Shuffle data
combined = np.column_stack((data, labels))
np.random.shuffle(combined)
data = combined.T[:-1].T
labels = combined.T[-1].T

# Split dataset
train_data = data[:-int(len(data) * test_split)]
train_labels = labels[:-int(len(labels) * test_split)]
test_data = data[-int(len(data) * test_split):]
test_labels = labels[-int(len(labels) * test_split):]

def batch_gen(batch_size, mode):
    if mode == 'train':
        input_data = train_data
        output_labels = train_labels
    elif mode == 'test':
        input_data = test_data
        output_labels = test_labels
    else:
        print "Invalid mode"
        exit(1)
    for i in range(0, len(input_data), batch_size):
        yield input_data[i:(i + batch_size)], output_labels[i:(i + batch_size)]

saver = tf.train.Saver()
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(num_epochs):
    if i == 0:
        patience = 0
        prev_loss = 0
    else:
        prev_loss = total_train_loss
    total_train_loss = 0
    total_train_f1_score = 0
    initial_time = time.time()
    for j, (batch_x, batch_y) in enumerate(batch_gen(batch_size, 
                                                     mode='train')):
        train_loss, train_f1s, _ = sess.run([loss, f1_score, train_step],
                                            feed_dict={x:batch_x,
                                            y_actual:batch_y})
        total_train_loss += train_loss
        total_train_f1_score += train_f1s
        if j % display_every == 0:
            time_left = ((time.time() - initial_time) / (j + 1)) * ((len(
                        train_labels) / batch_size) - (j + 1))
            sys.stdout.write("\rEpoch: %2d, Loss: %6.4f, F1 Score: %6.4f, "\
                             "ETA: %4ds" % (i + 1, total_train_loss / (j + 1),
                             total_train_f1_score / (j + 1), time_left))
            sys.stdout.flush()
    print "\rEpoch: %2d, Loss: %6.4f, F1 Score: %6.4f, Time Taken: %4ds" % (
          i + 1, total_train_loss / (j + 1), total_train_f1_score / (j + 1),
          time.time() - initial_time)
    if ((prev_loss - total_train_loss) / (j + 1)) < early_stop_threshold:
        patience += 1
        if patience >= early_stop_patience:
            break
    else:
        patience = 0
print "Saving model..."
saver.save(sess, './lstm_ad')
print "Saved model"

total_test_f1_score = 0
initial_time = time.time()
for j, (batch_x, batch_y) in enumerate(batch_gen(batch_size, mode='test')):
    test_f1s = sess.run(f1_score, feed_dict={x:batch_x, y_actual:batch_y})
    total_test_f1_score += test_f1s
    if j % display_every == 0:
        time_left = ((time.time() - initial_time) / (j + 1)) * ((len(
                    test_labels) / batch_size) - (j + 1))
        sys.stdout.write("\rTest F1 Score: %6.4f, ETA: %4ds" % (
                         total_test_f1_score / (j + 1), time_left))
        sys.stdout.flush()
print "\rTest F1 Score: %6.4f, Time Taken: %4ds" % (total_test_f1_score / (
      j + 1), time.time() - initial_time)

