import glob
import os
import pandas as pd
from regex import P


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


class Responses:
    ALL = "all"
    AVOID = "avoidance_response"
    CONFRONT = "confrontation_response"
    ESCAPE = "escape_response"
    SAFETY = "safety_behavior_response"


class Phases:
    BASE_REST = "Baseline_Rest"
    BASE_SPEECH = "Baseline_Speech"

    BUG_RELAX = "BugBox_Relax"
    BUG_INSTRUCTIONS = "BugBox_Instructions"
    BUG_ANTICIPATE = "BugBox_Anticipate"
    BUG_DECISION = "BugBox_Decision"
    BUG_EXPOSURE = "BugBox_Exposure"
    BUG_BREAK = "BugBox_Break"
    BUG_REFLECT = "BugBox_Reflect"

    SPEECH_RELAX = "Speech_Relax"
    SPEECH_INSTRUCTIONS = "Speech_Instructions"
    SPEECH_ANTICIPATE = "Speech_Anticipate"
    SPEECH_DECISION = "Speech_Decision"
    SPEECH_EXPOSURE = "Speech_Exposure"
    SPEECH_BREAK = "Speech_Break"
    SPEECH_REFLECT = "Speech_Reflect"

    PHASES_LIST = [
        BASE_REST,
        BASE_SPEECH,
        BUG_RELAX,
        BUG_INSTRUCTIONS,
        BUG_ANTICIPATE,
        BUG_DECISION,
        BUG_EXPOSURE,
        BUG_BREAK,
        BUG_REFLECT,
        SPEECH_RELAX,
        SPEECH_INSTRUCTIONS,
        SPEECH_ANTICIPATE,
        SPEECH_DECISION,
        SPEECH_EXPOSURE,
        SPEECH_BREAK,
        SPEECH_REFLECT
    ]

    BASELINE_PHASES = [
        BASE_REST,
        BASE_SPEECH
    ]

    BUG_PHASES = [
        BUG_RELAX,
        BUG_INSTRUCTIONS,
        BUG_ANTICIPATE,
        BUG_DECISION,
        BUG_EXPOSURE,
        BUG_BREAK,
        BUG_REFLECT
    ]

    SPEECH_PHASES = [
        SPEECH_RELAX,
        SPEECH_INSTRUCTIONS,
        SPEECH_ANTICIPATE,
        SPEECH_DECISION,
        SPEECH_EXPOSURE,
        SPEECH_BREAK,
        SPEECH_REFLECT
    ]


class Paths:
    # ROOT_DIR = os.path.abspath(os.path.join(PATH, os.pardir))
    ROOT_DIR = "C:\\Users\\zhoux\\Desktop\\Projects\\anxiety"
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    PARTICIPANT_DATA_DIR = os.path.join(DATA_DIR, "participants")
    METRICS = os.path.join(DATA_DIR, "metrics")

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
        data = pd.read_csv(f, header=None)
        # remove extra last column from ankle data
        if "LeftAnkle" in f or "RightAnkle" in f:
            data = data.iloc[:, :-1]
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
    df = pd.read_csv(file)
    p = df.iloc[index-1].loc[columns]
    return p


if __name__ == "__main__":
    pass
