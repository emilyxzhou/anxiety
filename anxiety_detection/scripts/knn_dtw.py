import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import samplerate
import tools.data_reader_apd as dr

from keras.utils import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier as Knn_Dtw


ROOT_DIR = "C:\\Users\\zhoux\\Desktop\\Projects\\anxiety"


def load_data(convert_sr, task, data_type, phase, unimodal=True):
    if unimodal:
        HA_BASELINES = np.vstack(HA_BASELINES)
        LA_BASELINES = np.vstack(LA_BASELINES)
        HA_BASELINES = np.transpose(HA_BASELINES)
        LA_BASELINES = np.transpose(LA_BASELINES)
    else:
        for arr in HA_BASELINES:
            print(arr.shape)
        HA_BASELINES = np.stack(HA_BASELINES, axis=2)
        LA_BASELINES = np.stack(LA_BASELINES, axis=2)
    HA_BASELINES = np.nan_to_num(HA_BASELINES, copy=False)
    LA_BASELINES = np.nan_to_num(LA_BASELINES, copy=False)
    num_samples_ha = HA_BASELINES.shape[1]
    num_samples_la = LA_BASELINES.shape[1]

    if convert_sr:
        HA_BASELINES = [samplerate.resample(HA_BASELINES[:, i], ratio=100.0 / 250.0) for i in range(num_samples_ha)]
        LA_BASELINES = [samplerate.resample(LA_BASELINES[:, i], ratio=100.0 / 250.0) for i in range(num_samples_la)]
        HA_BASELINES = np.vstack(HA_BASELINES)
        LA_BASELINES = np.vstack(LA_BASELINES)
        HA_BASELINES = np.transpose(HA_BASELINES)
        LA_BASELINES = np.transpose(LA_BASELINES)
        # print(HA_BASELINES.shape)
        # print(LA_BASELINES.shape)

    x_train = pad_sequences([HA_BASELINES, LA_BASELINES], padding="post")
    x_train = np.hstack([x_train[0, :, :], x_train[1, :, :]])
    x_train = np.transpose(x_train)

    # print(x_train.shape)
    x_train = x_train.astype(np.float32)
    y_train = [1 for _ in range(num_samples_ha)] + [0 for _ in range(num_samples_la)]
    y_train = np.asarray(y_train)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=16)
    return x_train, x_test, y_train, y_test


def train_knn_dtw(knn_dtw, x_train, y_train):
    """Train KNN-DTW"""
    knn_dtw.fit(x_train, y_train)
    return knn_dtw


def predict_knn_dtw(knn_dtw, x_test, y_test, verbose=True):
    preds = knn_dtw.predict_proba(x_test)
    preds = np.argmax(preds, axis=1)
    if verbose:
        print(f"Predictions: {preds}")
        print(f"Actual: {y_test}")
    return preds


def display_confusion_matrix(labels, preds, y_test):
    print(classification_report(
        preds, y_test, target_names=[l for l in labels.values()]
    ))

    conf_mat = confusion_matrix(preds, y_test)

    fig = plt.figure(figsize=(6, 6))

    res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
    for i, row in enumerate(conf_mat):
        for j, c in enumerate(row):
            if c > 0:
                plt.text(j - .2, i + .1, c, fontsize=16)

    cb = fig.colorbar(res)
    plt.title("Confusion Matrix")
    _ = plt.xticks(range(2), [l for l in labels.values()], rotation=90)
    _ = plt.yticks(range(2), [l for l in labels.values()])

    plt.show()


if __name__ == "__main__":
    # convert_sr = True
    convert_sr = False
    task = dr.Tasks.BUGS
    data_type = dr.DataTypes.ECG
    phase = dr.Phases.BUG_ANTICIPATE

    knn_dtw = Knn_Dtw()
    print("Loading data ...")
    x_train, x_test, y_train, y_test = load_data(
        convert_sr, task, data_type, phase,
        unimodal=True
    )
    labels = {0: "LA", 1: "HA"}

    # print(f"Training KNN-DTW on {data_type}_{phase} ...")
    # knn_dtw = train_knn_dtw(knn_dtw, x_train, y_train)
    # print("Running predictions on test set ...")
    # preds = predict_knn_dtw(knn_dtw, x_test, y_test)

    # display_confusion_matrix(labels, preds, y_test)
