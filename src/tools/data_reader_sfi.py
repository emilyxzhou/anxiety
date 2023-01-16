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


def get_participant_data(index):
    file = os.path.join(Paths.WESAD, f"S{index}", f"S{index}.pkl")
    data = pd.read_pickle(file)
    return data


CLIP_INDICES = list(range(1, 17))

def get_clip_timestamps(subject_index):
    if subject_index < 10:
        subject_index = "0" + str(subject_index)
    else:
        subject_index = str(subject_index)
    ts_file = os.path.join(Paths.SFI, f"VP{subject_index}", "Triggers.txt")
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


if __name__ == "__main__":
    ts_df = get_clip_timestamps(2)
    print(ts_df)

