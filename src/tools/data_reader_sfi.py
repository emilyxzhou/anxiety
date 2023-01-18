import glob
import math
import numpy as np
import os
import pandas as pd


class Paths:
    ROOT_DIR = "C:\\Users\\zhoux\\Desktop\\Projects\\anxiety"
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    SFI = os.path.join(DATA_DIR, "SFI")
    METRICS = os.path.join(DATA_DIR, "metrics", "SFI")


class Signals:
    BR = "BR"
    EDA = "GSR"
    ECG = "ECG"


def get_participant_data(index):
    file = os.path.join(Paths.WESAD, f"S{index}", f"S{index}.pkl")
    data = pd.read_pickle(file)
    return data


CLIP_INDICES = list(range(1, 17))
subject_folders = glob.glob(Paths.SFI + "/VP*")
SUBJECTS = [int(s.split("\\")[-1][-2:]) for s in subject_folders]


def get_clip_timestamps(subject):
    if subject < 10:
        subject = "0" + str(subject)
    else:
        subject = str(subject)
    ts_file = os.path.join(Paths.SFI, f"VP{subject}", "Triggers.txt")
    # df = pd.read_csv(ts_file)
    with open(ts_file) as f:
        stripped = (line.strip() for line in f)
        lines = [line.split("\t") for line in stripped if line]
    rows = [line[0] for line in lines]
    ts_data = [line[1:] for line in lines]
    df = pd.DataFrame(data=ts_data, columns=["start", "end"], index=rows)
    if "BIOFEEDBACK-OXYGEN-TRAININGS" in df.index:
        df = df.drop(index="BIOFEEDBACK-OXYGEN-TRAININGS")
    return df


def get_signal(subject, phases, signal):
    ts_df = get_clip_timestamps(subject)
    if subject < 10:
        subject = "0" + str(subject)
    else:
        subject = str(subject)
        
    if phases == "rest":
        phases = ["BIOFEEDBACK-REST"]
    elif phases == "all":
        phases = [f"CLIP-{str(phase)}" for phase in CLIP_INDICES] + ["BIOFEEDBACK-REST"]
    else:
        phases = [f"CLIP-{str(phase)}" for phase in phases]
        
    br_file = os.path.join(Paths.SFI, f"VP{subject}", f"Bitalino{signal}.txt")


    with open(br_file) as f:
        stripped = (line.strip() for line in f)
        lines = [line.split("\t") for line in stripped if line]
    lines = [line[:-1] for line in lines]
    df = pd.DataFrame(data=lines, columns=["signal", "timestamp"], dtype=float)

    out = {}
    for phase in phases:
        start_t = float(ts_df.loc[phase, "start"])
        end_t = float(ts_df.loc[phase, "end"])
        phase_data = df.loc[(df["timestamp"] >= start_t) & (df["timestamp"] < end_t)].loc[:, "signal"].reset_index(drop=True)
        out[phase] = phase_data
    # out = pd.DataFrame(data=out).transpose()
    return out


if __name__ == "__main__":
    data = get_signal(subject=2, phases=[1, 2, 3], signal=Signals.ECG)
