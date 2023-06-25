import glob
import numpy as np
import os
import pandas as pd


SUBJECTS = list(range(1, 31))
for i in range(len(SUBJECTS)):
    temp = str(SUBJECTS[i])
    SUBJECTS[i] = temp

CLIPS = [1, 2, 3, 4, 5, 6, 7, 8, 10]
# amusing: 1, 2
# boring: 3, 4
# relaxing: 5, 6
# scary: 7, 8
# 10: starting video; 11: in-between video; 12: ending video


class Signals:
    ECG = "ECG"  # column 2, 1000 Hz
    EDA = "EDA"  # column 4, 1000 Hz


class SelfReports:
    AROUSAL = "arousal"
    VALENCE = "valence"


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


def read_single_self_report(subject, clip, label_type=SelfReports.AROUSAL):
    """ subject: int """
    file = os.path.join(Paths.SELF_REPORTS, f"sub_{subject}.csv")
    data = pd.read_csv(file).loc[:, ["jstime", label_type, "video"]]
    data = data[data["video"] == clip]
    subject_col = [subject for _ in range(data.shape[0])]
    data.insert(0, "subject", subject_col)
    return data


def get_self_reports(label_type=SelfReports.AROUSAL):
    out = []
    for s in SUBJECTS:
        temp = [s]
        for c in CLIPS:
            data = read_single_self_report(s, c, label_type)
            temp.append(np.mean(data.loc[:, label_type]))
        out.append(temp)
    out = pd.DataFrame(out, columns=["subject"] + [c for c in CLIPS])
    return out

    
def get_mean_self_reports(label_type=SelfReports.AROUSAL):
    out = []
    for s in SUBJECTS:
        sum = 0
        for c in CLIPS:
            data = read_single_self_report(s, c, label_type)
            sum += np.mean(data.loc[:, label_type])
        mean = sum / len(CLIPS)
        out.append([s, mean])
    out = pd.DataFrame(out, columns=["subject", "mean"])
    return out


if __name__ == "__main__":
    data = get_self_reports(SelfReports.AROUSAL)
    print(data)