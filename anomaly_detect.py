#!/bin/python2
import tensorflow as tf
import numpy as np
import os
import sys
import time
import math
from operator import mul
from extract import *

program_name = 'anomaly_detect'

# Creates weights with the Xavier initializer
def weight_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()(shape)
    return tf.Variable(initial)

# Creates biases with the Xavier initializer
def bias_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()(shape)
    return tf.Variable(initial)

# Creates an LSTM layer
def LSTM(x, time_steps, hidden_nodes):
    x = tf.reshape(x, [-1, time_steps])
    x = tf.split(x, time_steps, axis=-1)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_nodes)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return outputs[-1]

# Creates two stacked LSTM layers
def LSTMx2(x, time_steps, hidden_nodes_1, hidden_nodes_2):
    x = tf.reshape(x, [-1, time_steps])
    x = tf.split(x, time_steps, axis=-1)
    lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(hidden_nodes_1)
    lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(hidden_nodes_2)
    lstm_stacked = tf.contrib.rnn.MultiRNNCell([lstm_cell1, lstm_cell2])
    outputs, states = tf.contrib.rnn.static_rnn(lstm_stacked, x,
                                                dtype=tf.float32)
    return outputs[-1]

# Creates a fully-connected linear layer
def dense(x, input_shape, num_neurons):
    if len(input_shape) > 2:
        flat = tf.reshape(x, [-1, reduce(mul, input_shape[1:], 1)])
    else:
        flat = x
    W = weight_variable([reduce(mul, input_shape[1:], 1), num_neurons])
    b = bias_variable([num_neurons])
    return tf.matmul(flat, W) + b

# Returns the probability density function for a normal distribution
def multivariate_normal_diag(x):
    mean, variance = tf.nn.moments(x, axes=0)
    #prob = tf.contrib.distributions.MultivariateNormalDiag(mean,
           #tf.sqrt(variance)).prob(x)
    pdf = tf.reduce_prod((1 / (tf.sqrt(2 * math.pi * variance))) * tf.pow(
          math.e, -1 * tf.divide(tf.squared_difference(x, mean),
          2 * variance)), axis=1)
    return pdf

# Returns the probability density function for a beta distribution
def beta_dist(x, features):
    alpha = tf.Variable(tf.random_uniform([features], minval=0, maxval=2))
    #alpha = tf.Variable(2.0 * tf.ones([features]))
    beta = tf.Variable(tf.random_uniform([features], minval=0, maxval=2))
    #beta = tf.Variable(2.0 * tf.ones([features]))
    beta_dist = tf.contrib.distributions.Beta(alpha, beta)
    return tf.reduce_mean(beta_dist.prob(x), axis=-1)

# Predicts classes w.r.t the threshold, using the probability density function
def predict_ad(prob, threshold):
    #cond = tf.less(prob, tf.scalar_mul(threshold,
                                          #tf.ones([tf.shape(prob)[0]])))
    cond = tf.greater(prob, tf.scalar_mul(threshold,
                                          tf.ones([tf.shape(prob)[0]])))
    out = tf.where(cond, tf.zeros(tf.shape(cond)), tf.ones(tf.shape(cond)))
    return out

def predict_soft(prob):
    threshold = tf.Variable(tf.random_uniform([], maxval=10))
    #scale = tf.Variable(tf.random_uniform([], minval=1, maxval=1e3))
    scale = 100
    return tf.nn.sigmoid((threshold - prob) * scale)

# Calculates the F1 score
def f1_score_func(actual, pred):
    #true_positives = tf.to_float(tf.count_nonzero(pred * actual))
    true_positives = tf.to_float(tf.reduce_sum(pred * actual))
    #false_positives = tf.to_float(tf.count_nonzero(pred * (actual - 1)))
    false_positives = tf.to_float(tf.reduce_sum(pred * (1 - actual)))
    #false_negatives = tf.to_float(tf.count_nonzero((pred - 1) * actual))
    false_negatives = tf.to_float(tf.reduce_sum((1 - pred) * actual))
    epsilon = tf.constant(1e-7, dtype=tf.float32)
    precision = tf.divide(true_positives,
                          true_positives + false_positives + epsilon)
    recall = tf.divide(true_positives,
                       true_positives + false_negatives + epsilon)
    return tf.divide(2 * precision * recall,\
                     precision + recall + epsilon)

x = tf.placeholder(tf.float32, [None, 3197])
y_actual = tf.placeholder(tf.float32, [None,])
lstm = LSTM(x, 3197, 128)
#lstm = LSTMx2(x, 3197, 128, 3197)
#y_pred = tf.squeeze(lstm, axis=-1)
#y_pred = dense(x, [None, 3197], 128)
#y_pred = multivariate_normal_diag(lstm)
#y_pred = multivariate_normal_diag(y_pred)
#y_pred = multivariate_normal_diag(lstm)
y_pred = beta_dist(tf.nn.sigmoid(lstm), 128)

#loss = tf.reduce_mean(tf.square(y_actual - y_pred))
#loss = tf.reduce_mean(tf.square(y_actual - tf.pow(math.e,
                                                  #-1 * tf.square(y_pred))))
pos_weight = tf.placeholder(tf.float32, [])
#loss = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(targets=y_actual,
                     #logits=1/y_pred, pos_weight=pos_weight))

threshold = tf.placeholder(tf.float32, [])
#prediction = predict(y_pred)
#prediction = predict_autoenc(x, y_pred)
#prediction = predict_ad(y_pred, threshold)
prediction = predict_soft(y_pred)
f1_score = f1_score_func(y_actual, prediction)
loss = 100 * (1 - f1_score)
train_step = tf.train.AdamOptimizer().minimize(loss)
#train_step = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9,
                                        #use_nesterov=True).minimize(loss)

# Train the model
def train_model(train_data=None, train_labels=None, test_data=None,
                test_labels=None, sess=None, test_split=0.3, num_epochs=100,
                batch_size=32, loss_weight=10, anomaly_threshold=5.0,
                early_stop_threshold=0.01, early_stop_patience=5,
                import_model=False, continue_train=True, display_every=1):
    if train_data is None or train_labels is None or test_data is None or\
    test_labels is None:
        train_data, train_labels, test_data, test_labels = split_data(
                                                           test_split=test_split)
    else:
        train_data, train_labels, test_data, test_labels = data_tup

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

    # Session and saver initialize
    saver = tf.train.Saver()
    if sess is None:
        if tf.get_default_session() is not None:
            sess = tf.get_default_session()
            if not continue_train:
                sess.close()
                sess = tf.InteractiveSession()
        else:
            sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    if import_model:
        print "Loading model..."
        saver = tf.train.import_meta_graph(program_name + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        print "Restored model"

    # Training and testing model
    try:
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
                                        feed_dict={x:batch_x, y_actual:batch_y,
                                        threshold:anomaly_threshold,
                                        pos_weight:loss_weight})
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

        # Test results
        total_test_f1_score = 0
        initial_time = time.time()
        for j, (batch_x, batch_y) in enumerate(batch_gen(batch_size, mode='test')):
            test_f1s = sess.run(f1_score, feed_dict={x:batch_x, y_actual:batch_y,
                            threshold:anomaly_threshold, pos_weight:loss_weight})
            total_test_f1_score += test_f1s
            if j % display_every == 0:
                time_left = ((time.time() - initial_time) / (j + 1)) * ((len(
                            test_labels) / batch_size) - (j + 1))
                sys.stdout.write("\rTest F1 Score: %6.4f, ETA: %4ds" % (
                                total_test_f1_score / (j + 1), time_left))
                sys.stdout.flush()
        print "\rTest F1 Score: %6.4f, Time Taken: %4ds" % (total_test_f1_score / (
            j + 1), time.time() - initial_time)
    except KeyboardInterrupt:
        print ''

    # Saving model
    response = raw_input("Do you want to save this model? (Y/n): ")
    if response.lower() not in ['n', 'no', 'nah', 'nein', 'nahi', 'nope']:
        saver.save(sess, './' + program_name)
        print "Saved model"

# Predict with the model
def predict_model(data, labels, sess=None, import_model=False,
                  anomaly_threshold=5.0, batch_size=32, display_every=1):
    default_sess_flag = tf.get_default_session() is not None
    if sess is None:
        if not import_model and default_sess_flag:
            sess = tf.get_default_session()
        else:
            if import_model and default_sess_flag:
                sess = tf.Session()
            else:
                sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())
            print "Loading model..."
            saver = tf.train.import_meta_graph(program_name + '.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./'))
            print "Restored model"

    def batch_gen(batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:(i + batch_size)], labels[i:(i + batch_size)]

    # Generate predictions
    total_f1_score = 0
    initial_time = time.time()
    results = []
    for j, (batch_x, batch_y) in enumerate(batch_gen(batch_size)):
        f1s, pred = sess.run([f1_score, prediction], feed_dict={x:batch_x,
                    y_actual:batch_y, threshold:anomaly_threshold})
        total_f1_score += f1s
        results += pred.tolist()
        if j % display_every == 0:
            time_left = ((time.time() - initial_time) / (j + 1)) * ((len(
                        labels) / batch_size) - (j + 1))
            sys.stdout.write("\rF1 Score: %6.4f, ETA: %4ds" % (
                            total_f1_score / (j + 1), time_left))
            sys.stdout.flush()
    print "\rF1 Score: %6.4f, Time Taken: %4ds" % (total_f1_score / (j + 1),
                                                   time.time() - initial_time)

    if import_model and default_sess_flag:
        sess.close()

    return np.array(results)

if __name__ == '__main__':
    train_model(anomaly_threshold=3, loss_weight=100, 
                early_stop_threshold=0.05)

