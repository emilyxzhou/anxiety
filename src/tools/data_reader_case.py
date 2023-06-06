import glob
import numpy as np
import os
import pandas as pd


SUBJECTS = list(range(1, 31))
for i in range(len(SUBJECTS)):
    temp = str(SUBJECTS[i])
    SUBJECTS[i] = temp

CLIPS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]
# amusing: 1, 2
# boring: 3, 4
# relaxing: 5, 6
# scary: 7, 8


class Signals:
    ECG = "ECG"  # column 2, 1000 Hz
    EDA = "EDA"  # column 4, 1000 Hz


class Paths:
    # ROOT_DIR = os.path.abspath(os.path.join(PATH, os.pardir))
    ROOT_DIR = "C:\\Users\\zhoux\\Desktop\\Projects\\anxiety"
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    METRICS = os.path.join(DATA_DIR, "metrics", "CASE")

    FILES = os.path.join(DATA_DIR, "CASE", "interpolated", "physiological") 
    SELF_REPORTS = os.path.join(DATA_DIR, "CASE", "interpolated", "annotations") 


def read_ecg(subject, clip):
    """ subject: int """
    file = os.path.join(Paths.FILES, f"sub_{subject}.csv")
    data = pd.read_csv(file).loc[:, ["daqtime", "ecg", "video"]]
    data = data[data["video"] == clip]
    subject_col = [subject for _ in range(data.shape[0])]
    data.insert(0, "subject", subject_col)
    data = data.drop("video", axis=1)
    return data


def read_gsr(subject, clip):
    """ subject: int """
    file = os.path.join(Paths.FILES, f"sub_{subject}.csv")
    data = pd.read_csv(file)
    data = pd.read_csv(file).loc[:, ["daqtime", "gsr", "video"]]
    data = data[data["video"] == clip]
    subject_col = [subject for _ in range(data.shape[0])]
    data.insert(0, "subject", subject_col)
    data = data.drop("video", axis=1)
    return data


def read_self_reports(subject, clip, self_report_type="arousal"):
    """ subject: int """
    file = os.path.join(Paths.SELF_REPORTS, f"sub_{subject}.csv")
    data = pd.read_csv(file).loc[:, ["jstime", self_report_type, "video"]]
    data = data[data["video"] == clip]
    subject_col = [subject for _ in range(data.shape[0])]
    data.insert(0, "subject", subject_col)
    return data


if __name__ == "__main__":
    subject = 1
    clip = 1
    data = read_ecg(subject, clip)
    print(data.head())