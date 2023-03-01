import glob
import numpy as np
import os
import pandas as pd


class Study1:
    BASELINE = "Baseline"
    POSITIVE_EMOTION_HIGH_APPROACH = "Positive_Emotion_High_Approach"
    THREAT = "Threat"
    NEUTRAL = "Neutral8"
    ALL = [BASELINE, POSITIVE_EMOTION_HIGH_APPROACH, THREAT, NEUTRAL]


class Study3:
    BASELINE = "Baseline"
    GRATITUDE = "Gratitude"
    NEUTRAL = "Neutral9"
    ALL = [BASELINE, GRATITUDE, NEUTRAL]


class Study4:
    BASELINE = "Baseline"
    FEAR = "Fear2"
    NEUTRAL = "Neutral10"
    ALL = [BASELINE, FEAR, NEUTRAL]


class Study5:
    ANGER1 = "Anger1"
    ANGER2 = "Anger2"
    ANGER3 = "Anger3"
    NEUTRAL1 = "Neutral1"
    NEUTRAL6 = "Neutral6"
    NEUTRAL7 = "Neutral7"
    AMUSEMENT2 = "Amusement2"
    AMUSEMENT3 = "Amusement3"
    AMUSEMENT4 = "Amusement4"
    ALL = [ANGER1, ANGER2, ANGER3, NEUTRAL1, NEUTRAL6, NEUTRAL7, AMUSEMENT2, AMUSEMENT3, AMUSEMENT4]


HIGH_AROUSAL = [
    Study1.POSITIVE_EMOTION_HIGH_APPROACH,
    Study1.THREAT,
    Study3.GRATITUDE,
    Study4.FEAR,
    Study5.ANGER1,
    Study5.ANGER2,
    Study5.ANGER3,
    Study5.AMUSEMENT2,
    Study5.AMUSEMENT3,
    Study5.AMUSEMENT4
]


HIGH_VALENCE = [
    Study1.POSITIVE_EMOTION_HIGH_APPROACH,
    Study5.AMUSEMENT2,
    Study5.AMUSEMENT3,
    Study5.AMUSEMENT4
]


class Signals:
    ECG = "ECG"
    EDA = "EDA"
    TEMP = "temp"
    RESP = "respiration"


class Paths:
    ROOT_DIR = "C:\\Users\\zhoux\\Desktop\\Projects\\anxiety"
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    POPANE = os.path.join(DATA_DIR, "POPANE")
    METRICS = os.path.join(DATA_DIR, "metrics", "POPANE")
    STUDY1 = os.path.join(POPANE, "Study1")
    STUDY3 = os.path.join(POPANE, "Study3")
    STUDY4 = os.path.join(POPANE, "Study4")
    STUDY5 = os.path.join(POPANE, "Study5")


def get_subjects(study_num):
    study_folder = os.path.join(Paths.POPANE, f"Study{study_num}/*")
    study_files = glob.glob(study_folder)
    subjects = np.sort(np.unique([int(file.split("\\")[-1].split("_")[1][1:]) for file in study_files]))
    return subjects


def get_data_for_subject(study_num, subject, phase, signal):
    study_folder = os.path.join(Paths.POPANE, f"Study{study_num}")
    file = os.path.join(study_folder, f"S{study_num}_P{subject}_{phase}.csv")
    df = pd.read_csv(file, skiprows=11)
    data = df[signal]
    return data


if __name__ == "__main__":
    data = get_data_for_subject(5, 285, Study5.ANGER2, Signals.ECG)
    print(data.head())