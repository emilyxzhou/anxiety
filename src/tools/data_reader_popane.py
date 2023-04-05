import glob
import numpy as np
import os
import pandas as pd


class Paths:
    ROOT_DIR = "C:\\Users\\zhoux\\Desktop\\Projects\\anxiety"
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    POPANE = os.path.join(DATA_DIR, "POPANE")
    METRICS = os.path.join(DATA_DIR, "metrics", "POPANE")
    STUDY1 = os.path.join(POPANE, "Study1")
    STUDY2 = os.path.join(POPANE, "Study2")
    STUDY3 = os.path.join(POPANE, "Study3")
    STUDY4 = os.path.join(POPANE, "Study4")
    STUDY5 = os.path.join(POPANE, "Study5")
    STUDY6 = os.path.join(POPANE, "Study6")
    STUDY7 = os.path.join(POPANE, "Study7")
    

class Study1:
    BASELINE = "Baseline"
    POSITIVE_EMOTION_HIGH_APPROACH = "Positive_Emotion_High_Approach"
    POSITIVE_EMOTION_LOW_APPROACH = "Positive_Emotion_Low_Approach"
    THREAT = "Threat"
    NEUTRAL = "Neutral8"
    ALL = [BASELINE, POSITIVE_EMOTION_HIGH_APPROACH, POSITIVE_EMOTION_LOW_APPROACH, THREAT, NEUTRAL]
    positive_emotion_subjects = sorted(set([float(f.split("_")[1][1:]) for f in glob.glob(os.path.join(Paths.STUDY1, f"*Positive*"))]))


class Study2:
    BASELINE = "Baseline"
    POSITIVE_EMOTION_HIGH_APPROACH = "Positive_Emotion_High_Approach"
    POSITIVE_EMOTION_LOW_APPROACH = "Positive_Emotion_Low_Approach"
    NEUTRAL = "Neutral8"
    THREAT = "Threat"
    ANGER = "Anger4"
    ALL = [BASELINE, POSITIVE_EMOTION_HIGH_APPROACH, POSITIVE_EMOTION_LOW_APPROACH, NEUTRAL, THREAT, ANGER]
    positive_emotion_subjects = sorted(set([float(f.split("_")[1][1:]) for f in glob.glob(os.path.join(Paths.STUDY2, f"*Positive*"))]))
    neutral_subjects = sorted(set([float(f.split("_")[1][1:]) for f in glob.glob(os.path.join(Paths.STUDY2, f"*Neutral*"))]))
    threat_subjects = sorted(set([float(f.split("_")[1][1:]) for f in glob.glob(os.path.join(Paths.STUDY2, f"*Threat*"))]))


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
    BASELINE = "Baseline"
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
    anger_subjects = sorted(set([float(f.split("_")[1][1:]) for f in glob.glob(os.path.join(Paths.STUDY5, f"*Anger*"))]))
    neutral_subjects = sorted(set([float(f.split("_")[1][1:]) for f in glob.glob(os.path.join(Paths.STUDY5, f"*Neutral*"))]))
    amusement_subjects = sorted(set([float(f.split("_")[1][1:]) for f in glob.glob(os.path.join(Paths.STUDY5, f"*Amusement*"))]))


class Study6:
    BASELINE = "Baseline"
    AMUSEMENT1 = "Amusement1"
    AMUSEMENT2 = "Amusement2"
    AMUSEMENT3 = "Amusement3"
    AMUSEMENT4 = "Amusement4"
    TENDERNESS1 = "Tenderness1"
    TENDERNESS2 = "Tenderness2"
    NEUTRAL1 = "Neutral1"
    NEUTRAL2 = "Neutral2"
    NEUTRAL3 = "Neutral3"
    NEUTRAL4 = "Neutral4"
    NEUTRAL5 = "Neutral5"
    NEUTRAL6 = "Neutral6"
    ANGER1 = "Anger1"
    ANGER2 = "Anger2"
    ANGER3 = "Anger3"
    DISGUST = "Disgust"
    FEAR = "Fear1"
    SADNESS = "Sadness1"
    ALL = [
        BASELINE, AMUSEMENT1, AMUSEMENT2, AMUSEMENT3, AMUSEMENT4, 
        TENDERNESS1, TENDERNESS2,
        NEUTRAL1, NEUTRAL2, NEUTRAL3, NEUTRAL4, NEUTRAL5, NEUTRAL6,
        ANGER1, ANGER2, ANGER3,
        DISGUST, FEAR, SADNESS
    ]


class Study7:
    BASELINE = "Baseline"
    AMUSEMENT = "Amusement4"
    ANGER = "Anger3"
    EXCITEMENT = "Excitement"
    NEUTRAL = "Neutral6"
    SADNESS = "Sadness2"
    ALL = [BASELINE, AMUSEMENT, ANGER, EXCITEMENT, NEUTRAL, SADNESS]


HIGH_AROUSAL = [
    Study1.POSITIVE_EMOTION_HIGH_APPROACH,
    Study1.THREAT,
    Study3.GRATITUDE,
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


def get_affect_labels(study, phases, threshold="fixed"):
    self_report_df = pd.read_csv(os.path.join(Paths.METRICS, study, "self_reports.csv"), index_col=0)
    popane_affect_label = []

    for i in range(self_report_df.shape[0]):
        if threshold == "fixed":
            mean = 5
        else:
            mean = np.nanmean(row)
        subject = self_report_df["subject"].iloc[i]
        row = self_report_df[phases].iloc[i].replace(-1, np.NaN)
        zero_idx = row.index[row < mean]
        one_idx = row.index[row >= mean]
        row[zero_idx] = 0
        row[one_idx] = 1
        row = pd.concat([pd.Series([subject], index=["subject"]), row])
        popane_affect_label.append(row)
        
    return pd.DataFrame(popane_affect_label, columns=["subject"] + phases)


if __name__ == "__main__":
    print(len(Study1.positive_emotion_subjects))
    print(Study1.positive_emotion_subjects)
    # print(Study2.positive_emotion_subjects)
    # print(Study2.threat_subjects)
    # print(Study5.anger_subjects)
    # print(Study5.amusement_subjects)
