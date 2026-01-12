import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score,\
    average_precision_score, accuracy_score
import math
tf.random.set_seed(2021)


def get_neigh_index(filename, max_neigh=None):
    neigh = np.loadtxt(filename, delimiter=',')
    neigh_index = []
    for i in range(len(neigh)):
        list_index = []
        for j in range(len(neigh[0])):
            if neigh[i][j] == 1:
                list_index.append(j)
        if max_neigh is not None:
            list_index = list_index[:max_neigh]
            if len(list_index) < max_neigh:
                list_index = list_index + [i] * (max_neigh - len(list_index))
        neigh_index.append(list_index)
    neigh_index = tf.cast(neigh_index, dtype=tf.int32)
    return neigh_index


def prepare_data(data, len_recent_time):
    data_recent = []
    for i in range(len(data) - len_recent_time):
        data_recent.append(data[i:i + len_recent_time])
    data_recent = tf.cast(np.array(data_recent), dtype=tf.float32)
    return data_recent


def loss_function(pred, y, dy_diff, a_dy=6, lambda_=0.005, epsilon=0.9, alpha=0.25, gamma=2):
    zeros = tf.zeros_like(pred, dtype=pred.dtype)
    focal_loss = -alpha * (1 - pred) ** gamma * y * tf.math.log(pred) - (1 - alpha) * pred ** gamma * (
                1 - y) * tf.math.log(1 - pred)
    focal_loss = tf.where(pred * y + (1 - pred) * (1 - y) > epsilon, zeros, focal_loss)
    focal_loss = tf.reduce_mean(focal_loss)

    dy_loss = tf.keras.losses.mean_absolute_error(dy_diff, 0.)
    dy_loss = a_dy - tf.reduce_mean(dy_loss)
    dy_loss = tf.reduce_max([0, dy_loss])

    loss = focal_loss + lambda_ * dy_loss
    return loss, focal_loss, dy_loss


def compute_loss(x, thre_nc, y_dy, y, model, batch_size, return_dy=False):
    batch_val = math.ceil(len(x) / batch_size)
    loss_mean = []
    for i in range(batch_val):
        y_pred, y_dy, dy_diff = model(x[i * batch_size:(i + 1) * batch_size],
                                      thre_nc[i * batch_size:(i + 1) * batch_size], y_dy)
        loss, focal_loss, dy_loss = loss_function(y_pred, y[i * batch_size:(i + 1) * batch_size], dy_diff)
        loss_mean.append(loss)
    if return_dy:
        return y_dy
    return tf.cast(np.array(loss_mean).mean(), dtype=tf.float32)


def get_f1_threshold(x, thre_nc, y_dy, y, model, batch_size):
    batch_val = math.ceil(len(x) / batch_size)
    val_pred = tf.zeros((batch_size, y.shape[-1]))
    for i in range(batch_val):
        y_pred, y_dy, dy_diff = model(x[i * batch_size:(i + 1) * batch_size],
                                      thre_nc[i * batch_size:(i + 1) * batch_size], y_dy)
        val_pred = tf.concat([val_pred, y_pred], axis=0)
    val_pred = val_pred[batch_size:]
    list1 = []
    list2 = []
    y = y.numpy().reshape((-1, 1))
    for j in np.arange(0, 1, 0.001):
        y_pred = (val_pred > j).numpy().reshape((-1, 1))
        f1 = f1_score(y, y_pred)
        accu = accuracy_score(y, y_pred)
        list1.append(f1)
        list2.append(accu)
    return np.arange(0, 1, 0.001)[np.argmax(list1)], np.arange(0, 1, 0.001)[np.argmax(list2)], y_dy


def get_metrics(x, thre_nc, y_dy, y, model, batch_size, threshold_f1, threshold_accu):
    batch_test = math.ceil(len(x) / batch_size)
    test_pred = tf.zeros((batch_size, y.shape[-1]))
    for i in range(batch_test):
        y_pred, y_dy, dy_diff = model(x[i * batch_size:(i + 1) * batch_size],
                                      thre_nc[i * batch_size:(i + 1) * batch_size], y_dy)
        test_pred = tf.concat([test_pred, y_pred], axis=0)
    test_pred = test_pred[batch_size:].numpy().reshape((-1, 1))
    y = y.numpy().reshape((-1, 1))
    ap_score = average_precision_score(y, test_pred)
    ra_score = roc_auc_score(y, test_pred)
    y_pred_f1 = (test_pred > threshold_f1)
    y_pred_accu = (test_pred > threshold_accu)
    f1 = np.max([f1_score(y, y_pred_f1), f1_score(y, y_pred_accu)])
    recall = recall_score(y, y_pred_f1)
    precision = precision_score(y, y_pred_f1, zero_division=0)
    accu = np.max([accuracy_score(y, y_pred_f1), accuracy_score(y, y_pred_accu)])
    return ap_score, ra_score, f1, recall, precision, accu, y, test_pred


def get_threshold_max_precision(y_true, y_prob, min_recall, step=0.001):
    y_true = np.asarray(y_true).reshape((-1, 1))
    y_prob = np.asarray(y_prob).reshape((-1, 1))
    best_th = 0.5
    best_prec = -1.0
    best_recall = -1.0
    for th in np.arange(0.0, 1.0, step):
        y_pred = (y_prob > th)
        rec = recall_score(y_true, y_pred)
        if rec + 1e-12 < float(min_recall):
            continue
        prec = precision_score(y_true, y_pred, zero_division=0)
        if prec > best_prec:
            best_prec = float(prec)
            best_recall = float(rec)
            best_th = float(th)
    return best_th, best_prec, best_recall


def get_threshold_max_recall(y_true, y_prob, min_precision=None, min_accuracy=None, step=0.001):
    y_true = np.asarray(y_true).reshape((-1, 1))
    y_prob = np.asarray(y_prob).reshape((-1, 1))

    best_th = 0.5
    best_recall = -1.0
    best_prec = -1.0
    best_acc = -1.0

    for th in np.arange(0.0, 1.0, step):
        y_pred = (y_prob > th)

        prec = float(precision_score(y_true, y_pred, zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))
        if (min_precision is not None) and (prec + 1e-12 < float(min_precision)):
            continue
        if (min_accuracy is not None) and (acc + 1e-12 < float(min_accuracy)):
            continue

        rec = float(recall_score(y_true, y_pred))
        if (rec > best_recall) or (
            abs(rec - best_recall) < 1e-12 and prec > best_prec
        ):
            best_recall = rec
            best_prec = prec
            best_acc = acc
            best_th = float(th)

    return best_th, best_recall, best_prec, best_acc


def streaming_postprocess(test_pred, alpha=0.8, th_on=0.5, th_off=0.45):
    test_pred = np.asarray(test_pred).reshape((-1, 1)).astype(np.float32)
    th_on = float(th_on)
    th_off = float(th_off)
    if th_off > th_on:
        th_off = th_on

    smooth = test_pred
    state = np.zeros_like(test_pred, dtype=bool)
    prev_state = False
    for i in range(test_pred.shape[0]):
        p = float(test_pred[i, 0])
        if prev_state:
            prev_state = p >= th_off
        else:
            prev_state = p >= th_on
        state[i, 0] = prev_state
    return smooth, state


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
                            early_stop = EarlyStopping(patience=10,delta=0.000001)
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, current_val_loss):
        current_score = current_val_loss

        if self.best_score is None:
            self.best_score = current_score

        elif current_score > self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            print(f'EarlyStopping update val_loss: {self.best_score} --> {current_score}')
            self.best_score = current_score
            self.counter = 0
