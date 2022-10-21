import glob
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


def get_participant_data(index):
    file = os.path.join(Paths.WESAD, f"S{index}", f"S{index}.pkl")
    data = pd.read_pickle(file)
    return data


def get_modality(index, location, modality):
    data = get_participant_data(index)
    return data["signal"][location][modality]


if __name__ == "__main__":
    ecg_chest = get_modality(2, "chest", Modalities.ECG)

    print(ecg_chest.shape)
