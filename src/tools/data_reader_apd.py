import glob
import numpy as np
import os
import pandas as pd

# NOTE: SUBJECT 8 DID NOT DO THE BUG EXPOSURE PHASE
# NOT ALL test_phases CAN BE USED -- NO SELF-REPORT GIVEN FOR SOME PHASES


class Tasks:
    ALL = "all"
    BASELINE = "baseline"
    BUGS = "bug_box_task"
    SPEAKING = "speaking_task"


class DataTypes:
    ANKLE_L = "LeftAnkle"
    ANKLE_R = "RightAnkle"
    WRIST_L = "LeftWrist"
    WRIST_R = "RightWrist"
    EDA = "EDA"
    ECG = "Heart"
    POSTURE = "Posture"

    DATA_TYPES = [
        ANKLE_L, ANKLE_R, WRIST_L, WRIST_R, EDA, ECG, POSTURE
    ]


class Groups: 
    ALL = "all"
    HA = "high_anxiety_group"
    LA = "low_anxiety_group"

    ha_participant_indices = [
        'P4', 'P6', 'P7', 'P8', 'P10', 'P12', 'P15', 'P16', 'P18', 'P22', 'P26', 'P27', 'P29', 'P31', 'P32', 'P33', 'P35', 'P42', 'P45', 'P47', 'P48', 'P49', 'P54', 'P55', 'P66', 'P69'
    ]

    la_participant_indices = [
        'P14', 'P21', 'P23', 'P25', 'P34', 'P39', 'P43', 'P46', 'P51', 'P57', 'P71', 'P72', 'P77', 'P78', 'P79', 'P80', 'P82', 'P83', 'P84', 'P85', 'P87', 'P88', 'P89', 'P91', 'P92', 'P93'
    ]


class Responses:
    ALL = "all"
    AVOID = "avoidance_response"
    CONFRONT = "confrontation_response"
    ESCAPE = "escape_response"
    SAFETY = "safety_behavior_response"


class Phases:
    BASE_REST = "Baseline_Rest"
    # BASE_SPEECH = "Baseline_Speech"

    BUG_RELAX = "BugBox_Relax"
    # BUG_INSTRUCTIONS = "BugBox_Instructions"
    BUG_ANTICIPATE = "BugBox_Anticipate"
    # BUG_DECISION = "BugBox_Decision"
    BUG_EXPOSURE = "BugBox_Exposure"
    BUG_BREAK = "BugBox_Break"
    # BUG_REFLECT = "BugBox_Reflect"

    SPEECH_RELAX = "Speech_Relax"
    # SPEECH_INSTRUCTIONS = "Speech_Instructions"
    SPEECH_ANTICIPATE = "Speech_Anticipate"
    # SPEECH_DECISION = "Speech_Decision"
    SPEECH_EXPOSURE = "Speech_Exposure"
    SPEECH_BREAK = "Speech_Break"
    # SPEECH_REFLECT = "Speech_Reflect"

    PHASES_LIST = [
        BASE_REST,
        # BASE_SPEECH,
        BUG_RELAX,
        # BUG_INSTRUCTIONS,
        BUG_ANTICIPATE,
        # BUG_DECISION,
        BUG_EXPOSURE,
        BUG_BREAK,
        # BUG_REFLECT,
        SPEECH_RELAX,
        # SPEECH_INSTRUCTIONS,
        SPEECH_ANTICIPATE,
        # SPEECH_DECISION,
        SPEECH_EXPOSURE,
        SPEECH_BREAK,
        # SPEECH_REFLECT
    ]

    BASELINE_PHASES = [
        BASE_REST,
        # BASE_SPEECH
    ]

    BUG_PHASES = [
        BUG_RELAX,
        # BUG_INSTRUCTIONS,
        BUG_ANTICIPATE,
        # BUG_DECISION,
        BUG_EXPOSURE,
        BUG_BREAK,
        # BUG_REFLECT
    ]

    SPEECH_PHASES = [
        SPEECH_RELAX,
        # SPEECH_INSTRUCTIONS,
        SPEECH_ANTICIPATE,
        # SPEECH_DECISION,
        SPEECH_EXPOSURE,
        SPEECH_BREAK,
        # SPEECH_REFLECT
    ]

    # NOTE: NO SELF-REPORTS AVAILABLE FOR 'BASE SPEECH' AND 'REFLECT' PHASES
    phases = {
        # "Baseline": [BASE_REST, BASE_SPEECH],
        "Baseline": [BASE_REST],
        "Bug baseline": [BUG_RELAX],
        "Speech baseline": [SPEECH_RELAX],
        # "Bug all": [BUG_RELAX, BUG_ANTICIPATE, BUG_EXPOSURE, BUG_BREAK, BUG_REFLECT],
        "Bug all": [BUG_RELAX, BUG_ANTICIPATE, BUG_EXPOSURE, BUG_BREAK],
        # "Speech all": [SPEECH_RELAX, SPEECH_ANTICIPATE, SPEECH_EXPOSURE, SPEECH_BREAK, SPEECH_REFLECT],
        "Speech all": [SPEECH_RELAX, SPEECH_ANTICIPATE, SPEECH_EXPOSURE, SPEECH_BREAK],
        "Bug pre-anxiety": [BUG_RELAX, BUG_ANTICIPATE],
        "Speech pre-anxiety": [SPEECH_RELAX, SPEECH_ANTICIPATE],
        "Bug anxiety": [BUG_EXPOSURE],
        "Speech anxiety": [SPEECH_EXPOSURE],
        # "Bug post-anxiety": [BUG_BREAK, BUG_REFLECT],
        "Bug post-anxiety": [BUG_BREAK],
        # "Speech post-anxiety": [SPEECH_BREAK, SPEECH_REFLECT],
        "Speech post-anxiety": [SPEECH_BREAK],
    }

    test_phases = [
        phases["Baseline"],  # 0
        phases["Bug baseline"],
        phases["Speech baseline"],
        phases["Bug baseline"] + phases["Speech baseline"],
        phases["Baseline"] + phases["Bug baseline"],
        phases["Baseline"] + phases["Speech baseline"],
        phases["Baseline"] + phases["Bug baseline"] + phases["Speech baseline"],

        phases["Bug all"],  # 7
        phases["Speech all"],
        phases["Bug all"] + phases["Speech all"],
        phases["Baseline"] + phases["Bug all"],
        phases["Baseline"] + phases["Speech all"],
        phases["Baseline"] + phases["Bug all"] + phases["Speech all"],

        phases["Bug pre-anxiety"],  # 13
        phases["Speech pre-anxiety"],
        phases["Bug pre-anxiety"] + phases["Speech pre-anxiety"],
        phases["Baseline"] + phases["Bug pre-anxiety"],
        phases["Baseline"] + phases["Speech pre-anxiety"],
        phases["Baseline"] + phases["Bug pre-anxiety"] + phases["Speech pre-anxiety"],

        phases["Bug anxiety"],  # 19
        phases["Speech anxiety"],
        phases["Bug pre-anxiety"] + phases["Speech anxiety"],
        phases["Baseline"] + phases["Bug anxiety"],
        phases["Baseline"] + phases["Speech anxiety"],
        phases["Baseline"] + phases["Bug anxiety"] + phases["Speech anxiety"],

        phases["Bug post-anxiety"],  # 25
        phases["Speech post-anxiety"],
        phases["Bug post-anxiety"] + phases["Speech post-anxiety"],
        phases["Baseline"] + phases["Bug post-anxiety"],
        phases["Baseline"] + phases["Speech post-anxiety"],
        phases["Baseline"] + phases["Bug post-anxiety"] + phases["Speech post-anxiety"],

        phases["Bug pre-anxiety"] + phases["Bug anxiety"],  # 31
        phases["Speech pre-anxiety"] + phases["Speech anxiety"],
        phases["Bug pre-anxiety"] + phases["Bug anxiety"] + phases["Speech pre-anxiety"] + phases["Speech anxiety"],
        phases["Baseline"] + phases["Bug pre-anxiety"] + phases["Bug anxiety"],
        phases["Baseline"] + phases["Speech pre-anxiety"] + phases["Speech anxiety"],
        phases["Baseline"] + phases["Bug pre-anxiety"] + phases["Bug anxiety"] + phases["Speech pre-anxiety"] + phases["Speech anxiety"],

        phases["Bug post-anxiety"] + phases["Bug anxiety"],  # 37
        phases["Speech post-anxiety"] + phases["Speech anxiety"],
        phases["Bug post-anxiety"] + phases["Bug anxiety"] + phases["Speech post-anxiety"] + phases["Speech anxiety"],
        phases["Baseline"] + phases["Bug post-anxiety"] + phases["Bug anxiety"],
        phases["Baseline"] + phases["Speech post-anxiety"] + phases["Speech anxiety"],
        phases["Baseline"] + phases["Bug post-anxiety"] + phases["Bug anxiety"] + phases["Speech post-anxiety"] + phases["Speech anxiety"],

        phases["Bug pre-anxiety"] + phases["Bug post-anxiety"],  # 43
        phases["Speech pre-anxiety"] + phases["Speech post-anxiety"],
        phases["Bug pre-anxiety"] + phases["Bug post-anxiety"] + phases["Speech pre-anxiety"] + phases["Speech post-anxiety"],
        phases["Baseline"] + phases["Bug pre-anxiety"] + phases["Bug post-anxiety"],
        phases["Baseline"] + phases["Speech pre-anxiety"] + phases["Speech post-anxiety"],
        phases["Baseline"] + phases["Bug pre-anxiety"] + phases["Bug post-anxiety"] + phases["Speech pre-anxiety"] + phases["Speech post-anxiety"],
    ]


class Paths:
    # ROOT_DIR = os.path.abspath(os.path.join(PATH, os.pardir))
    ROOT_DIR = "C:\\Users\\zhoux\\Desktop\\Projects\\anxiety"
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    PARTICIPANT_DATA_DIR = os.path.join(DATA_DIR, "participants")
    METRICS = os.path.join(DATA_DIR, "metrics", "APD")

    ANKLE_MOVEMENT = os.path.join(DATA_DIR, "ankle_movement_data")
    ECG = os.path.join(DATA_DIR, "electrocardiogram_data")
    GSR = os.path.join(DATA_DIR, "electrodermal_data")  # electrodermal activity/EDA/GSR
    TORSO = os.path.join(DATA_DIR, "torso_posture_and_activity_data")
    WRIST = os.path.join(DATA_DIR, "wrist_activity_data")


def get_data(data_type, phase):
    if data_type not in DataTypes.DATA_TYPES:
        raise ValueError(f"Invalid data type: {data_type}")
    if phase not in Phases.PHASES_LIST:
        raise ValueError(f"Invalid phase: {phase}")


def read_ankle_movement(task="all", group="all", response="all"):
    data_folder = Paths.ANKLE_MOVEMENT
    files = get_requested_folders(data_folder, task, group, response)
    dfs = get_dataframes_from_files(files)
    return dfs


def read_ecg(task="all", group="all", response="all"):
    data_folder = Paths.ECG
    files = get_requested_folders(data_folder, task, group, response)
    dfs = get_dataframes_from_files(files)
    return dfs


def read_gsr(task="all", group="all", response="all"):
    data_folder = Paths.GSR
    files = get_requested_folders(data_folder, task, group, response)
    dfs = get_dataframes_from_files(files)
    return dfs


def read_torso(task="all", group="all", response="all"):
    data_folder = Paths.TORSO
    files = get_requested_folders(data_folder, task, group, response)
    dfs = get_dataframes_from_files(files)
    return dfs


def read_wrist(task="all", group="all", response="all"):
    data_folder = Paths.WRIST
    files = get_requested_folders(data_folder, task, group, response)
    dfs = get_dataframes_from_files(files)
    return dfs


def get_requested_folders(data_folder, task, group, response):
    files = []
    if task == "all" and group == "all" and response == "all":
        files = glob.glob(data_folder + "/*/*/*/*.csv")
    elif task != "all" and group == "all" and response == "all":
        files = glob.glob(os.path.join(data_folder, task, "*", "*", "*.csv"))
    elif task != "all" and group != "all" and response == "all":
        files = glob.glob(f"{data_folder}/{task}/{group}/*/*.csv")
    elif task != "all" and group == "all" and response != "all":
        files = glob.glob(os.path.join(data_folder, task, "*", response, "*.csv"))
    elif task != "all" and group != "all" and response != "all":
        files = glob.glob(os.path.join(data_folder, task, group, response, "*.csv"))
    elif task == "all" and group != "all" and response == "all":
        files = glob.glob(os.path.join(data_folder, "*", group, "*", "*.csv"))
    elif task == "all" and group == "all" and response != "all":
        files = glob.glob(os.path.join(data_folder, "*", "*", response, "*.csv"))
    elif task == "all" and group != "all" and response != "all":
        files = glob.glob(os.path.join(data_folder, "*", group, response, "*.csv"))
    else:
        print("Invalid request")
    if len(files) == 0:
        print("No files found, returning empty list")
    return files


def get_participant_folder(index):
    return glob.glob(Paths.PARTICIPANT_DATA_DIR + f"/*/p_{index}")


def get_dataframes_from_files(files):
    """
    :param files: list of .csv file paths to construct dataframes from
    :return: list of pd.Dataframes}
    """
    dfs = []
    for f in files:
        subject = f.split("\\")[-3].split("_")[-1]
        data = pd.read_csv(f, header=None, skip_blank_lines=True, low_memory=False)
        # remove extra last column from ankle data
        if "LeftAnkle" in f or "RightAnkle" in f:
            data = data.iloc[:, :-1]
        num_rows = data.shape[0]
        if num_rows <= 1:
            data = pd.DataFrame({"subject": [subject]*2, "0": [0]*2, "1": [0]*2})
            # print(data.shape)
            # print(data.head())
        else: 
            data.insert(0, "subject", pd.array([subject for _ in range(num_rows)]))
        dfs.append(data)
    return dfs


def _get_participant_index_from_file(file_path):
    file_name = file_path.split(os.sep)[-1]
    file_info = file_name.split("_")
    participant = file_info[0]
    index = participant[1:]
    return index


def _get_data_type_from_file(file_path):
    file_name = file_path.split(os.sep)[-1]
    file_info = file_name.split("_")
    data_type = file_info[1].split(".")[0]
    return data_type


def get_participant_dataframes_from_index(index, task, group):
    """
    :param index: index of participant to construct
    :param task: task name (bug_box_task or speaking_task)
    :param group: anxiety level (high_anxiety_group or low_anxiety_group)
    :return: dictionary in the format {"{data_type}_{participant_index}": pd.Dataframe} for
        the specified participant
    """
    files = glob.glob(Paths.DATA_DIR + f"/*/{task}/{group}/*/P{index}_*.csv")
    if len(files) == 0:
        print(f"No files found for participant {index}, {group}.")
        return None
    dfs = get_dataframes_from_files(files)
    return dfs


def get_participant_details(index, columns):
    # print(f"Retrieving participant {index}'s {columns} data...")
    file = os.path.join(Paths.DATA_DIR, "participants_details.csv")
    df = pd.read_csv(file, skip_blank_lines=True)
    p = df.iloc[index-1].loc[columns]
    return p


def get_gender_labels():
    participant_file = os.path.join(Paths.DATA_DIR, "participants_details.csv")
    df = pd.read_csv(participant_file)
    gender_df = df.loc[:, ["Participant", "Gender"]]

    ha_gender_df = gender_df.loc[gender_df['Participant'].isin(Groups.ha_participant_indices)]
    la_gender_df = gender_df.loc[gender_df['Participant'].isin(Groups.la_participant_indices)]

    indices_m = ha_gender_df.index[ha_gender_df["Gender"] == "M"]
    indices_f = ha_gender_df.index[ha_gender_df["Gender"] == "F"]
    temp_ha = ha_gender_df.loc[:, "Gender"].copy()
    temp_ha.loc[indices_m] = 0
    temp_ha.loc[indices_f] = 1
    temp_ha = temp_ha.astype(int).reset_index(drop=True)

    indices_m = la_gender_df.index[la_gender_df["Gender"] == "M"]
    indices_f = la_gender_df.index[la_gender_df["Gender"] == "F"]
    temp_la = la_gender_df.loc[:, "Gender"].copy()
    temp_la.loc[indices_m] = 0
    temp_la.loc[indices_f] = 1

    ha_gender_df = ha_gender_df.drop("Gender", axis=1)
    la_gender_df = la_gender_df.drop("Gender", axis=1)
    ha_gender_df["Gender"] = temp_ha
    la_gender_df["Gender"] = temp_la
    temp_la = temp_la.astype(int).reset_index(drop=True)

    labels = pd.concat([temp_ha, temp_la])

    return labels

def get_dass_labels(dass="Anxiety", threshold="fixed"):
    participant_file = os.path.join(Paths.DATA_DIR, "participants_details.csv")
    df = pd.read_csv(participant_file)
    if dass == "Anxiety":
        label = f"DASS_{dass}_Score"
    else:
        label = f"DASS_{dass}_Scores"
    dass_labels = df.loc[:, ["Participant", label]]
    mean_dass = np.mean(dass_labels.loc[:, label])

    ha_dass_df = dass_labels.loc[dass_labels['Participant'].isin(Groups.ha_participant_indices)]
    la_dass_df = dass_labels.loc[dass_labels['Participant'].isin(Groups.la_participant_indices)]

    if threshold == "fixed":
        if dass == "Anxiety":
            ha_indices_h = ha_dass_df.index[ha_dass_df[label] >= 10]
            ha_indices_l = ha_dass_df.index[ha_dass_df[label] < 10]
            la_indices_h = la_dass_df.index[la_dass_df[label] >= 10]
            la_indices_l = la_dass_df.index[la_dass_df[label] < 10]
        else:
            ha_indices_h = ha_dass_df.index[ha_dass_df[label] >= 14]
            ha_indices_l = ha_dass_df.index[ha_dass_df[label] < 14]
            la_indices_h = la_dass_df.index[la_dass_df[label] >= 14]
            la_indices_l = la_dass_df.index[la_dass_df[label] < 14]
    else:
        ha_indices_h = ha_dass_df.index[ha_dass_df[label] >= mean_dass]
        ha_indices_l = ha_dass_df.index[ha_dass_df[label] < mean_dass]
        la_indices_h = la_dass_df.index[la_dass_df[label] >= mean_dass]
        la_indices_l = la_dass_df.index[la_dass_df[label] < mean_dass]

    temp_ha = ha_dass_df.loc[:, label].copy()
    temp_ha.loc[ha_indices_h] = 1
    temp_ha.loc[ha_indices_l] = 0
    temp_ha = temp_ha.astype(int).reset_index(drop=True)

    temp_la = la_dass_df.loc[:, label].copy()
    temp_la.loc[la_indices_h] = 1
    temp_la.loc[la_indices_l] = 0
    temp_la = temp_la.astype(int).reset_index(drop=True)

    ha_dass_df = ha_dass_df.drop(label, axis=1)
    la_dass_df = la_dass_df.drop(label, axis=1)
    ha_dass_df[label] = temp_ha
    la_dass_df[label] = temp_la

    labels = pd.concat([temp_ha, temp_la])

    return labels

def get_suds_labels(threshold="fixed"):
    participant_file = os.path.join(Paths.DATA_DIR, "participants_details.csv")
    df = pd.read_csv(participant_file)
    labels = [
        "Baseline_SUDS", "BugBox_Relax_SUDS", "BugBox_Preparation_SUDS", "BugBox_Exposure_SUDS", "BugBox_Exposure_SUDS", 
        "Speech_Relax_SUDS", "Speech_SUDS", "Speech_Exposure_SUDS", "Speech_Break_SUDS"
    ]
    suds_labels = df.loc[:, ["Participant"] + labels]
    mean_suds = np.mean(suds_labels.loc[:, labels])

    ha_suds_df = suds_labels.loc[suds_labels['Participant'].isin(Groups.ha_participant_indices)]
    la_suds_df = suds_labels.loc[suds_labels['Participant'].isin(Groups.la_participant_indices)]

    if threshold == "fixed":
        indices_h = ha_suds_df.index[ha_suds_df[label] >= 10]
        indices_l = ha_dass_df.index[ha_dass_df[label] < 10]
    else:
        indices_h = ha_dass_df.index[ha_dass_df[label] >= mean_dass]
        indices_l = ha_dass_df.index[ha_dass_df[label] < mean_dass]

    temp_ha = ha_dass_df.loc[:, label].copy()
    temp_ha.loc[indices_h] = 1
    temp_ha.loc[indices_l] = 0
    temp_ha = temp_ha.astype(int).reset_index(drop=True)

    indices_h = la_dass_df.index[la_dass_df[label] >= mean_dass]
    indices_l = la_dass_df.index[la_dass_df[label] < mean_dass]
    temp_la = la_dass_df.loc[:, label].copy()
    temp_la.loc[indices_h] = 1
    temp_la.loc[indices_l] = 0
    temp_la = temp_la.astype(int).reset_index(drop=True)

    ha_dass_df = ha_dass_df.drop(label, axis=1)
    la_dass_df = la_dass_df.drop(label, axis=1)
    ha_dass_df[label] = temp_ha
    la_dass_df[label] = temp_la

    labels = pd.concat([temp_ha, temp_la])

    return labels


if __name__ == "__main__":
    pass
