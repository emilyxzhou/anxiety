import datetime
import numpy as np
import os
import pandas as pd

from tools import data_reader as dr
from tools import display_tools as dt


def create_participant_files(indices=None, exist_ok=True):
    HA_FOLDER = os.path.join(dr.Paths.PARTICIPANT_DATA_DIR, dr.Groups.HA)
    LA_FOLDER = os.path.join(dr.Paths.PARTICIPANT_DATA_DIR, dr.Groups.LA)
    if not os.path.isdir(HA_FOLDER):
        os.makedirs(HA_FOLDER)
    if not os.path.isdir(LA_FOLDER):
        os.makedirs(LA_FOLDER)
    if indices is None:
        indices = list(range(1, 95))
    for i in indices:
        p_data = dr.get_participant_dataframes_from_index(i, dr.Tasks.BUGS, dr.Groups.HA)
        p_group = dr.Groups.HA
        if p_data is None:
            p_data = dr.get_participant_dataframes_from_index(i, dr.Tasks.SPEAKING, dr.Groups.HA)
        if p_data is None:
            p_data = dr.get_participant_dataframes_from_index(i, dr.Tasks.BUGS, dr.Groups.LA)
            p_group = dr.Groups.LA
        if p_data is None:
            p_data = dr.get_participant_dataframes_from_index(i, dr.Tasks.SPEAKING, dr.Groups.LA)
            p_group = dr.Groups.LA
        if p_data is None:
            print(f"Participant {i} data does not exist.")
            continue
        print(f"Participant {i} data loaded. {p_group}.")

        # create participant folders if not exists
        P_FOLDER = os.path.join(dr.Paths.PARTICIPANT_DATA_DIR, p_group, f"p_{i}")
        if not os.path.isdir(P_FOLDER):
            os.makedirs(P_FOLDER)
        else:
            if not exist_ok:
                print(f"Data folder for participant {i} already exists, skipping")
                continue
            else:
                print(f"Data folder for participant {i} already exists, but overwriting")
        if p_data is not None:
            path = os.path.join(P_FOLDER, "baseline")
            if not os.path.isdir(path):
                os.makedirs(path)
            path = os.path.join(P_FOLDER, "bug_box_task")
            if not os.path.isdir(path):
                os.makedirs(path)
            path = os.path.join(P_FOLDER, "speaking_task")
            if not os.path.isdir(path):
                os.makedirs(path)
            p_data = label_participant_data(p_data)
            keys = list(p_data.keys())
            # for j in range(len(keys)):
            #     print(keys[j])
            #     print(p_data[keys[j]].shape)
            #     print(p_data[keys[j]].head())
            data_types = [key.split("_")[0] for key in keys]
            for j in range(len(keys)):
                with open(os.path.join(P_FOLDER, f"{data_types[j]}.csv"), "w+") as f:
                    p_data[keys[j]].to_csv(f, index=False)


def label_participant_data(p_data):
    keys = list(p_data.keys())
    for key in keys:
        data_type = key.split("_")[0]
        if data_type in ["LeftAnkle", "RightAnkle", "LeftWrist", "RightWrist"]:
            p_data[key].columns = [
                "timestamp", "a_x", "a_y", "a_z", "w_x", "w_y", "w_z", "roll", "yaw", "pitch"
            ]
        elif data_type == "Heart":
            p_data[key].columns = ["timestamp", "ECG reading"]
        elif data_type == "EDA":
            p_data[key].columns = ["timestamp", "Grove sensor reading"]
        elif data_type == "Posture":
            p_data[key].columns = ["timestamp", "degrees", "activity level (VMU)"]
    return p_data


def separate_full_dataset_into_phases(indices=None, verbose=False):
    if indices is None:
        indices = list(range(1, 95))
    for i in indices:
        print(f"Participant {i}")
        try:
            P_FOLDER = dr.get_participant_folder(i)[0]
        except IndexError:
            print(f"Participant {i} does not exist.")
            continue
        group = P_FOLDER.split("\\")[-2]
        print(group)
        # if not os.path.isdir(P_FOLDER):
        if len(os.listdir(P_FOLDER)) == 0:
            print(f"Participant {i} has no data, skipping.")
            continue

        base_phases = dr.Phases.BASELINE_PHASES.copy()
        bug_phases = dr.Phases.BUG_PHASES.copy()
        speech_phases = dr.Phases.SPEECH_PHASES.copy()
        bug_start = dr.get_participant_details(i, [f"{dr.Phases.BUG_RELAX}_Timestamp"])[0]
        speech_start = dr.get_participant_details(i, [f"{dr.Phases.SPEECH_RELAX}_Timestamp"])[0]

        if bug_start < speech_start:
            base_phases.append(dr.Phases.BUG_RELAX)
            bug_phases.append(dr.Phases.SPEECH_RELAX)
            speech_phases.append("END")
        else:
            base_phases.append(dr.Phases.SPEECH_RELAX)
            speech_phases.append(dr.Phases.BUG_RELAX)
            bug_phases.append("END")

        # print(base_phases)
        # print(bug_phases)
        # print(speech_phases)

        for data_type in [
            dr.DataTypes.ANKLE_L, dr.DataTypes.ANKLE_R,
            dr.DataTypes.WRIST_L, dr.DataTypes.WRIST_R,
            dr.DataTypes.EDA, dr.DataTypes.ECG, dr.DataTypes.POSTURE
        ]:
            f = os.path.join(P_FOLDER, f"{data_type}.csv")
            df = pd.read_csv(f)
            print(f"DATA TYPE: {data_type} ------------------------------")

            # BASELINE PHASES
            print("Processing baseline phases...")
            for j in range(len(base_phases)-1):
                if verbose: print(f"Phase: {base_phases[j]} to {base_phases[j+1]}")
                file = os.path.join(P_FOLDER, "baseline", f"{data_type}_{base_phases[j]}.csv")
                start_t = dr.get_participant_details(i, [f"{base_phases[j]}_Timestamp"])[0]
                end_t = dr.get_participant_details(i, [f"{base_phases[j+1]}_Timestamp"])[0]
                if verbose: print(f"Phase start: {start_t}, end: {end_t}\nDuration: {(end_t-start_t)/1000.0}s")
                phase_df = df.loc[(start_t <= df["timestamp"])]
                phase_df = phase_df.loc[(phase_df["timestamp"] <= end_t)]
                with open(file, "w+") as f:
                    phase_df.to_csv(f, index=False)

            # BUG BOX PHASES
            print("Processing bug box phases...")
            for j in range(len(bug_phases)-1):
                start_t = dr.get_participant_details(i, [f"{bug_phases[j]}_Timestamp"])[0]
                file = os.path.join(P_FOLDER, "bug_box_task", f"{data_type}_{bug_phases[j]}.csv")
                if bug_phases[j+1] == "END":
                    if verbose: print(f"Phase: {bug_phases[j]} to END")
                    phase_df = df.loc[(start_t <= df["timestamp"])]
                    with open(file, "w+") as f:
                        phase_df.to_csv(f, index=False)
                else:
                    if verbose: print(f"Phase: {bug_phases[j]} to {bug_phases[j+1]}")
                    end_t = dr.get_participant_details(i, [f"{bug_phases[j+1]}_Timestamp"])[0]
                    if verbose: print(f"Phase start: {start_t}, end: {end_t}\nDuration: {(end_t-start_t)/1000.0}s")
                    phase_df = df.loc[(start_t <= df["timestamp"])]
                    phase_df = phase_df.loc[(phase_df["timestamp"] <= end_t)]
                    with open(file, "w+") as f:
                        phase_df.to_csv(f, index=False)

            # SPEECH PHASES
            print("Processing speech phases...")
            for j in range(len(speech_phases)-1):
                start_t = dr.get_participant_details(i, [f"{speech_phases[j]}_Timestamp"])[0]
                file = os.path.join(P_FOLDER, "speaking_task", f"{data_type}_{speech_phases[j]}.csv")
                if speech_phases[j+1] == "END":
                    if verbose: print(f"Phase: {speech_phases[j]} to END")
                    phase_df = df.loc[(start_t <= df["timestamp"])]
                    with open(file, "w+") as f:
                        phase_df.to_csv(f, index=False)
                else:
                    if verbose: print(f"Phase: {speech_phases[j]} to {speech_phases[j+1]}")
                    end_t = dr.get_participant_details(i, [f"{speech_phases[j+1]}_Timestamp"])[0]
                    if verbose: print(f"Phase start: {start_t}, end: {end_t}\nDuration: {(end_t-start_t)/1000.0}s")
                    phase_df = df.loc[(start_t <= df["timestamp"])]
                    phase_df = phase_df.loc[(phase_df["timestamp"] <= end_t)]
                    with open(file, "w+") as f:
                        phase_df.to_csv(f, index=False)


if __name__ == "__main__":
    pass
    # create_participant_files(indices=[], exist_ok=True)
    # separate_full_dataset_into_phases()
