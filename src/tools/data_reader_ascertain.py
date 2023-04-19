import glob
import numpy as np
import os
import pandas as pd
import scipy.io as sio


SUBJECTS = list(range(1, 59))
for i in range(len(SUBJECTS)):
    temp = str(SUBJECTS[i])
    if len(temp) == 1:
        temp = "0" + temp
    SUBJECTS[i] = temp
CLIPS = [str(i) for i in list(range(1, 37))]


class SelfReports:
    AROUSAL = "Arousal"
    VALENCE = "Valence"
    ENGAGEMENT = "Engagement"
    LIKING = "Liking"
    FAMILIARITY = "Familiarity"
    SELF_REPORTS = [
        AROUSAL, VALENCE, ENGAGEMENT, LIKING, FAMILIARITY
    ]


class Paths:
    # ROOT_DIR = os.path.abspath(os.path.join(PATH, os.pardir))
    ROOT_DIR = "C:\\Users\\zhoux\\Desktop\\Projects\\anxiety"
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    METRICS = os.path.join(DATA_DIR, "metrics", "ASCERTAIN")

    ECG = os.path.join(DATA_DIR, "ASCERTAIN", "ECGData")  # 256 Hz
    GSR = os.path.join(DATA_DIR, "ASCERTAIN", "GSRData")  # 128 Hz


def read_ecg(subject, clip, axis="left"):
    # subject: 2-digit string for the subject number (01, 02, 14, 15, etc)
    # clip: string for the clip number (1, 2, 14, 15, etc)
    # axis: "left", "right", corresponds to the signal measured from the arm
    # Returns the corresponding column of ECG data as a pd.DataFrame
    file = os.path.join(Paths.ECG, f"Movie_P{subject}", f"ECG_Clip{clip}.mat")
    mat = sio.loadmat(file)["Data_ECG"]
    if axis == "right":
        df = pd.DataFrame(data=mat[:, -2], columns=["ecg"])
        df["subject"] = int(subject)
        df["clip"] = int(clip)
    elif axis == "left":
        df = pd.DataFrame(data=mat[:, -1], columns=["ecg"])
        df["subject"] = int(subject)
        df["clip"] = int(clip)
    df = df[["subject", "clip", "ecg"]]
    return df


def read_gsr(subject, clip, axis="x"):
    # subject: 2-digit string for the subject number (01, 02, 14, 15, etc)
    # clip: string for the clip number (1, 2, 14, 15, etc)
    # axis = "x", "y", "z", corresponds to the accelerometer axis
    # Returns the corresponding column of ECG data as a pd.DataFrame
    file = os.path.join(Paths.GSR, f"Movie_P{subject}", f"GSR_Clip{clip}.mat")
    mat = sio.loadmat(file)["Data_GSR"]
    if axis == "x":
        df = pd.DataFrame(data=mat[:, 2], columns=["eda"])
        df["subject"] = int(subject)
        df["clip"] = int(clip)
    elif axis == "y":
        df = pd.DataFrame(data=mat[:, 3], columns=["eda"])
        df["subject"] = int(subject)
        df["clip"] = int(clip)
    else: # axis == "z"
        df = pd.DataFrame(data=mat[:, 4], columns=["eda"])
        df["subject"] = int(subject)
        df["clip"] = int(clip)
    df = df[["subject", "clip", "eda"]]
    return df


def get_self_reports(self_report_type):
    if self_report_type not in SelfReports.SELF_REPORTS:
        raise NameError(f"{self_report_type} is not a valid self-report type")
    
    file = os.path.join(Paths.DATA_DIR, "ASCERTAIN", "Dt_SelfReports.mat")
    mat = sio.loadmat(file)["Ratings"]
    idx = SelfReports.SELF_REPORTS.index(self_report_type)
    df = pd.DataFrame(data=mat[idx, :, :], columns=CLIPS)
    df["subject"] = list(range(1, 59))
    df = df[["subject"] + [clip for clip in CLIPS]]
    return df


def get_mean_self_reports(self_report_type):
    self_report_df = get_self_reports(self_report_type)
    # mean_self_reports = pd.DataFrame(self_report_df[CLIPS].mean(axis=1), columns=["mean"])
    mean_self_reports = pd.DataFrame(self_report_df[CLIPS].median(axis=1), columns=["mean"])  # use median because self-reports are strongly skewed to the right
    mean_self_reports.insert(0, column="subject", value=list(range(1, 59)))
    return mean_self_reports
    


if __name__ == "__main__":
    # for p in ["01", "02"]:
    #     ecg = read_ecg(p, "1")
    #     print(ecg)
    #     eda = read_gsr(p, "1")
    #     print(eda)

    # self_reports = get_self_reports("Arousal")
    # print(self_reports.head())

    mean_self_reports = get_mean_self_reports("Arousal")
    print(mean_self_reports)
