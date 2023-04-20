import glob
import math
import numpy as np
import os
import pandas as pd


class Paths:
    ROOT_DIR = "C:\\Users\\zhoux\\Desktop\\Projects\\anxiety"
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    WESAD = os.path.join(DATA_DIR, "WESAD")
    METRICS = os.path.join(DATA_DIR, "metrics", "WESAD")


class Signals:
    ACC = "ACC"  # chest and wrist
    ECG = "ECG"  # chest only
    EDA = "EDA"  # chest and wrist
    EMG = "EMG"  # chest only
    RESP = "RESP"  # chest only
    TEMP = "TEMP"  # chest and wrist
    BVP = "BVP"  # wrist only


class WESADKeys:
    SUBJECT = "subject"
    SIGNAL = "signal"
    LABEL = "label"
    
    CHEST = "chest"
    WRIST = "wrist"


subject_indices = list(range(2, 12)) + list(range(13, 18))
SUBJECTS = [str(i) for i in subject_indices]


FS_DICT = {
    "chest": {
        "ACC": 700,
        "ECG": 700,
        "EDA": 700,
        "EMG": 700,
        "RESP": 700,
        "TEMP": 700
    },
    "wrist": {
        "ACC": 32,
        "BVP": 64,
        "EDA": 4,
        "TEMP": 4,
    }
}


class Phases:
    BASE = "Base"
    TSST = "TSST"
    MEDI_1 = "Medi_1"
    FUN = "Fun"
    MEDI_2 = "Medi_2"
    PHASE_ORDER = [BASE, TSST, MEDI_1, FUN, MEDI_2]

    def get_index(phase):
        return Phases.PHASE_ORDER.index(phase)


def get_participant_data(index):
    file = os.path.join(Paths.WESAD, f"S{index}", f"S{index}.pkl")
    data = pd.read_pickle(file)
    return data


def get_data_for_phase(index, phase, location, modality):
    data = get_modality(index, location, modality)
    start, end = get_time_intervals(index, phase)
    fs = FS_DICT[location][modality]
    start_index = math.floor(start*fs)
    end_index = math.ceil(end*fs)
    return data[start_index:end_index, :].flatten()


def get_modality(index, location, modality):
    data = get_participant_data(index)
    return data[WESADKeys.SIGNAL][location][modality]


def get_self_reports(phases, index, type):
    file = os.path.join(Paths.WESAD, f"S{index}", f"S{index}_quest.csv")
    df = pd.read_csv(file, sep=";", header=None, index_col=0).dropna(how="all")
    data = df.loc[f"# {type}", :].dropna(how="all", axis=1).transpose()
    columns = df.loc[f"# ORDER", :].dropna(how="all").tolist()[0:5]

    data = data.set_axis(columns, axis=1)
    data = data.rename(columns={"Medi 1": "Medi_1", "Medi 2": "Medi_2"})
    data = data.reindex(labels=Phases.PHASE_ORDER, axis=1)
    
    for col in data.columns:
        if col not in phases:
            data = data.drop(labels=col, axis=1)

    return data


def get_stai_scores(phases):
    self_report_type = "STAI"
    # phases = Phases.PHASE_ORDER
    columns = [f"{phase}_STAI" for phase in phases]
    subjects = SUBJECTS

    stai_scores = []

    for s in subjects:
        stai = get_self_reports(phases, s, self_report_type)
        stai = stai.astype(int)
        for i in range(stai.shape[1]):
            stai.iloc[0, i] = 5 - stai.iloc[0, i]
            stai.iloc[3, i] = 5 - stai.iloc[3, i]
            stai.iloc[5, i] = 5 - stai.iloc[5, i]
        stai = stai.sum(axis=0)/6*20  # proper scaling
        stai = stai.tolist()
        stai.insert(0, int(s))
        stai_scores.append(stai)
    stai_scores = pd.DataFrame(data=stai_scores, columns=["subject"] + columns)
    return stai_scores

def get_stai_labels(phases, threshold="fixed"):
    stai_scores = get_stai_scores(phases)
    columns = ["subject"] + [f"{phase}_STAI" for phase in Phases.PHASE_ORDER]

    y_labels = []
    for i in range(stai_scores.shape[0]):
        if threshold != "fixed":
            label_mean = stai_scores.iloc[i, 1:].mean()
        else:
            label_mean = 50
        labels = [stai_scores.iloc[i, 0]]  # subject ID
        for j in range(1, stai_scores.shape[1]):
            if stai_scores.iloc[i, j] < label_mean:
                labels.append(0)
            else:
                labels.append(1)
        y_labels.append(labels)
    y_labels = pd.DataFrame(data=y_labels, columns=columns)
    return y_labels


def get_dim_scores(phases, dim_type="valence"):
    if dim_type == "valence":
        index = 0
    else:
        index = 1
    self_report_type = "DIM"
    # phases = Phases.PHASE_ORDER
    columns = [f"{phase}_{dim_type}" for phase in phases]
    subjects = SUBJECTS

    dim_scores = []

    for s in subjects:
        dim = get_self_reports(phases, s, self_report_type).iloc[index, :]
        dim = dim.astype(int)
        dim = dim.tolist()
        dim.insert(0, int(s))
        dim_scores.append(dim)
    dim_scores = pd.DataFrame(data=dim_scores, columns=["subject"] + columns)
    return dim_scores


def get_time_intervals(index, phase):
    file = os.path.join(Paths.WESAD, f"S{index}", f"S{index}_quest.csv")
    df = pd.read_csv(file, sep=";", header=None, index_col=0).dropna(how="all")
    start = df.loc["# START", :].iloc[Phases.get_index(phase)]
    end = df.loc["# END", :].iloc[Phases.get_index(phase)]
    return [float(start), float(end)]


if __name__ == "__main__":
    # ecg_chest_2 = get_modality(2, WESADKeys.CHEST, Modalities.ECG)
    # print(ecg_chest_2.shape)
    responses_2 = get_self_reports(2)
