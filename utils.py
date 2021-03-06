import datetime
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score


def print_curtime(note=None):
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    if note is not None:
        print(f"{note}: {current_time}")
    else:
        print(f"Current time: {current_time}")


def tf_allow_growth():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def create_logdir(root="logs/", args=None):
    log_dir = root + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        print(args, file=f)
    print(f"LOG_DIR: {log_dir}")
    summary_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, "train"))
    summary_writer.set_as_default()
    return log_dir


def auc_score(y_true, y_pred):
    if len(np.unique(y_true[:, 0])) == 1:
        return 0.5
    else:
        return roc_auc_score(y_true, y_pred)


def auc(y_true, y_pred):
    return tf.numpy_function(auc_score, (y_true, y_pred), tf.double)


def num_params(model):
    total_parameters = 0
    embed_parameters = 0
    dense_parameters = 0
    for variable in model.trainable_variables:
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters
        if 'embedding' in variable.name:
            embed_parameters += variable_parameters
        else:
            dense_parameters += variable_parameters

    print(f"Total Params: {total_parameters}")
    print(f"Dense Params: {dense_parameters}")
    print(f"Embed Params: {embed_parameters}")

    return total_parameters
