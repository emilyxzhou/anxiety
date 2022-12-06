import glob
import math
import numpy as np
import os
import pandas as pd


class Paths:
    ROOT_DIR = "C:\\Users\\zhoux\\Desktop\\Projects\\anxiety"
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    WESAD = os.path.join(DATA_DIR, "WESAD")


class Modalities:
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
    MEDI_1 = "Medi 1"
    FUN = "Fun"
    MEDI_2 = "Medi 2"
    PHASE_ORDER = [BASE, TSST, MEDI_1, FUN, MEDI_2]

    def get_index(phase):
        return Phases.PHASE_ORDER.index(phase)


def get_participant_data(index):
    file = os.path.join(Paths.WESAD, f"S{index}", f"S{index}.pkl")
    data = pd.read_pickle(file)
    return data


def get_data_for_phase(index, phase, location, modality):
    data = get_modality(index, location, modality)
    print(data.shape)
    start, end = get_time_intervals(index, phase)
    fs = FS_DICT[location][modality]
    start_index = math.floor(start*fs)
    end_index = math.ceil(end*fs)
    return data[start_index:end_index, :].flatten()


def get_modality(index, location, modality):
    data = get_participant_data(index)
    return data[WESADKeys.SIGNAL][location][modality]


def get_self_reports(index, type, phase):
    file = os.path.join(Paths.WESAD, f"S{index}", f"S{index}_quest.csv")
    df = pd.read_csv(file, sep=";", header=None, index_col=0).dropna(how="all")
    data = df.loc[f"# {type}", :].iloc[Phases.get_index(phase), :]
    return data


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
