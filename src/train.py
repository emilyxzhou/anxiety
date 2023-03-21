import glob
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import random
import shap
import scipy.signal as ss

import tools.data_reader_apd as dr_a
import tools.data_reader_popane as dr_p
import tools.data_reader_sfi as dr_s
import tools.data_reader_wesad as dr_w
import tools.preprocessing as preprocessing

from scipy.fft import fft, fftfreq, fftshift
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score, \
    recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.preprocessing import normalize

import cvxopt.solvers
cvxopt.solvers.options['show_progress'] = False

import warnings
warnings.filterwarnings(
    "ignore", 
    category=RuntimeWarning
)


class Metrics:
    BPM = "bpm"
    RMSSD = "rmssd"
    HF_RR = "hf_rr"
    LF_RR = "lf_rr"
    IBI=  "ibi"
    SDNN = "sdnn"
    MEAN_SCL = "mean_SCL"
    SCR_RATE = "SCR_rate"
    RESP = "breathingrate"
    MEAN_ANKLE_ACT_L = "mean_ankle_activity_l"
    MEAN_ANKLE_ACT_R =  "mean_ankle_activity_r"
    MEAN_WRIST_ACT = "mean_wrist_activity"
    MEAN_WRIST_ACT_L = "mean_wrist_activity_l"
    MEAN_WRIST_ACT_R =  "mean_wrist_activity_r"
    PEAK_ANKLE_ACC_L = "peak_ankle_acc_l"
    PEAK_ANKLE_ACC_R = "peak_ankle_acc_r"
    PEAK_WRIST_ACC = "peak_wrist_acc"
    PEAK_WRIST_ACC_L = "peak_wrist_acc_l"
    PEAK_WRIST_ACC_R = "peak_wrist_acc_r"
    MEAN_POSTURE = "mean_posture"

    ALL = [
        BPM, RMSSD, HF_RR, LF_RR, IBI, SDNN,
        MEAN_SCL, SCR_RATE,
        RESP,
        MEAN_ANKLE_ACT_L, MEAN_ANKLE_ACT_R,
        MEAN_WRIST_ACT_L, MEAN_WRIST_ACT_R,
        PEAK_ANKLE_ACC_L, PEAK_ANKLE_ACC_R,
        PEAK_WRIST_ACC_L, PEAK_WRIST_ACC_R,
        MEAN_POSTURE
    ]

    ECG = [BPM, RMSSD, HF_RR, LF_RR, IBI, SDNN]
    EDA = [MEAN_SCL, SCR_RATE]
    ANKLE = [MEAN_ANKLE_ACT_L, MEAN_ANKLE_ACT_R, PEAK_ANKLE_ACC_L, PEAK_ANKLE_ACC_R]
    WRIST = [MEAN_WRIST_ACT_L, MEAN_WRIST_ACT_R, PEAK_WRIST_ACC_L, PEAK_WRIST_ACC_R]


def train_test_split(x, y, test_size=0.15, by_subject=True):
    if by_subject:
        subjects = list(x.loc[:, "subject"].unique())
        indices = random.sample(subjects, int(len(subjects)*test_size))
        # print(f"test subjects: {indices}")
        x_train = x[~x["subject"].isin(indices)]
        y_train = y[~y["subject"].isin(indices)]
        x_test = x[x["subject"].isin(indices)]
        y_test = y[y["subject"].isin(indices)]
    else:
        num_samples = x.shape[0]
        indices = random.sample(range(num_samples), int(num_samples*test_size))
        x_train = x[~x.index.isin(indices)]
        y_train = y[~y.index.isin(indices)]
        x_test = x[x.index.isin(indices)]
        y_test = y[y.index.isin(indices)]

    return x_train, y_train, x_test, y_test, indices


def train_predict(models, x, y, test_size=0.15, by_subject=True, save_metrics=True, get_shap_values=False):
    """
    models: dictionary of {"name": model}
    """
    out = {}
    x_train, y_train, x_test, y_test, test_subjects = train_test_split(x, y, test_size, by_subject)
    while y_test.loc[:, "label"].nunique() == 1:
        print("Only one label in test data, rerunning train_test_split")
        x_train, y_train, x_test, y_test, test_subjects = train_test_split(x, y, test_size, by_subject)
    # print(f"x_train: {x_train.shape}")
    # print(f"y_train: {y_train.shape}")
    y_test = y_test.loc[:, "label"]
    for model_name in models.keys():
        model = models[model_name]
        model.fit(x_train, y_train.loc[:, "label"])
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        if save_metrics:
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred)
            report = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc
            }
        else:
            report = None
        if get_shap_values and acc > 0.65:
            try: 
                explainer = shap.Explainer(model)
            except Exception as e:
                explainer = shap.Explainer(model.predict, x_train)
            shap_values = explainer(x_test)
        else:
            shap_values = None
        out[model_name] = (acc, report, shap_values)
    return out


class Train_APD:
    
    def get_ratings():
        SUDS_labels = [
            "Participant",
            "Baseline_SUDS",
            "BugBox_Relax_SUDS", "BugBox_Preparation_SUDS", "BugBox_Exposure_SUDS", "BugBox_Break_SUDS",
            "Speech_Relax_SUDS", "Speech_SUDS", "Speech_Exposure_SUDS", "Speech_Break_SUDS"
        ]

        ha_participant_indices = [
            'P4', 'P6', 'P7', 'P8', 'P10', 'P12', 'P15', 'P16', 'P18', 'P22', 'P26', 'P27', 'P29', 'P31', 'P32', 'P33', 'P35', 'P42', 'P45', 'P47', 'P48', 'P49', 'P54', 'P55', 'P66', 'P69'
        ]

        la_participant_indices = [
            'P14', 'P21', 'P23', 'P25', 'P34', 'P39', 'P43', 'P46', 'P51', 'P57', 'P71', 'P72', 'P77', 'P78', 'P79', 'P80', 'P82', 'P83', 'P84', 'P85', 'P87', 'P88', 'P89', 'P91', 'P92', 'P93'
        ]

        participant_file = os.path.join(dr_a.Paths.DATA_DIR, "participants_details.csv")
        df = pd.read_csv(participant_file)

        suds_df = df[SUDS_labels]
        ha_suds_df = suds_df.loc[suds_df['Participant'].isin(ha_participant_indices)]
        la_suds_df = suds_df.loc[suds_df['Participant'].isin(la_participant_indices)]

        ha_suds_df = ha_suds_df.rename(columns={"Participant": "subject"})
        la_suds_df = la_suds_df.rename(columns={"Participant": "subject"})

        for i in range(ha_suds_df.shape[0]):
            p = int(ha_suds_df.iloc[i, ha_suds_df.columns.get_loc("subject")][1:])
            ha_suds_df.iloc[i, ha_suds_df.columns.get_loc("subject")] = p
        for i in range(la_suds_df.shape[0]):
            p = int(la_suds_df.iloc[i, la_suds_df.columns.get_loc("subject")][1:])
            la_suds_df.iloc[i, la_suds_df.columns.get_loc("subject")] = p

        # ha_suds_df['median'] = ha_suds_df.iloc[:, 1:].median(axis=1)
        # la_suds_df['median'] = la_suds_df.iloc[:, 1:].median(axis=1)
        ha_suds_df['median'] = ha_suds_df.iloc[:, 1:].median(axis=1)
        la_suds_df['median'] = la_suds_df.iloc[:, 1:].median(axis=1)
        columns = {c: SUDS_labels.index(c)-1 for c in ha_suds_df.columns[1:-1]}

        ha_rankings = ha_suds_df.rename(columns={c: SUDS_labels.index(c)-1 for c in ha_suds_df.columns[1:-1]}).reset_index(drop=True)
        la_rankings = la_suds_df.rename(columns={c: SUDS_labels.index(c)-1 for c in la_suds_df.columns[1:-1]}).reset_index(drop=True)

        return ha_rankings, la_rankings


    def get_apd_data_ranking(metrics, phases, verbose=False, anxiety_label_type=None, threshold="fixed"):
        """
        anxiety_label_type: can be None, "Trait", "Anxiety", "Depression", "Gender", "Random"
        """
        metrics_folder = dr_a.Paths.METRICS
        ha_rankings, la_rankings = Train_APD.get_ratings()

        columns = metrics.copy()
        columns.insert(0, "subject")

        data_x = []
        data_y = pd.concat([ha_rankings, la_rankings], axis=0).reset_index(drop=True)

        for phase in phases:
            if verbose: print(f"Generating features for phase {phase} " + "-"*30)
            phase_id = phases.index(phase)
            ha_features = []
            la_features = []

            for i in range(len(metrics)):
                metric = metrics[i]
                if verbose: print(f"Generating features for metric {metric}")
                file = os.path.join(metrics_folder, f"{metric}_{phase}_ha.csv")
                arr = pd.read_csv(file, index_col=[0]).to_numpy()

                if i == 0:  # subject IDs
                    ids = np.reshape(arr[:, 0], (arr[:, 0].size, 1))
                    ids = pd.DataFrame(data=ids, columns=["subject"])
                    ha_features.append(ids)

                # arr = arr[1:, 1:]
                col_mean = np.nanmean(arr, axis=1)
                idx = np.where(np.isnan(arr))
                arr[idx] = np.take(col_mean, idx[0])
                arr = np.nan_to_num(arr)
                arr = np.mean(arr[:, 1:], axis=1)
                arr = np.reshape(arr, (arr.size, 1))
                arr = pd.DataFrame(data=arr, columns=[f"{metric}"])
                ha_features.append(arr)

                file = os.path.join(metrics_folder, f"{metric}_{phase}_la.csv")
                arr = pd.read_csv(file, index_col=[0]).to_numpy()

                if i == 0:  # subject IDs
                    ids = np.reshape(arr[:, 0], (arr[:, 0].size, 1))
                    ids = pd.DataFrame(data=ids, columns=["subject"])
                    la_features.append(ids)

                # arr = arr[1:, 1:]
                col_mean = np.nanmean(arr, axis=1)
                idx = np.where(np.isnan(arr))
                arr[idx] = np.take(col_mean, idx[0])
                arr = np.nan_to_num(arr)
                arr = np.mean(arr[:, 1:], axis=1)
                arr = np.reshape(arr, (arr.size, 1))
                arr = pd.DataFrame(data=arr, columns=[f"{metric}"])
                la_features.append(arr)

            if anxiety_label_type is not None: 
                if anxiety_label_type == "Trait":
                    ha_group = pd.DataFrame(data=[1 for _ in range(len(ha_features[0]))])
                    la_group = pd.DataFrame(data=[0 for _ in range(len(la_features[0]))])
                    anxiety_label = pd.concat([ha_group, la_group])
                elif anxiety_label_type == "Anxiety":
                    anxiety_label = dr_a.get_dass_labels("Anxiety", threshold)
                elif anxiety_label_type == "Depression":
                    anxiety_label = dr_a.get_dass_labels("Depression", threshold)
                elif anxiety_label_type == "Gender":
                    anxiety_label = dr_a.get_gender_labels()
                    anxiety_label = dr_a.get_gender_labels()

            ha_features = pd.concat(ha_features, axis=1)
            la_features = pd.concat(la_features, axis=1)
            x = pd.concat([ha_features, la_features], axis=0)
            # print(x["subject"].value_counts().iloc[0:8])
            phase = pd.DataFrame(data=[phase_id for _ in range(x.shape[0])])

            x.insert(1, "phaseId", phase)

            if anxiety_label_type is not None: 
                x.insert(1, "anxietyGroup", anxiety_label)

            data_x.append(x)
        
        data_x = pd.concat(data_x).reset_index(drop=True)
        # data_x.sort_values(by=["phaseId", "subject"], inplace=True)

        # print(data_x.head())
        # print(data_y.head())

        subjects = data_x.loc[:, "subject"]
        phase_col = data_x.loc[:, "phaseId"]
        label = []
        for i in range(data_x.shape[0]):
            s = int(subjects.iloc[i])
            p = int(phase_col.iloc[i])
            rating = data_y.loc[data_y["subject"] == s].loc[:, p].values[0]
            med = data_y.loc[data_y["subject"] == s].loc[:, 'median'].values[0]
            if rating < med:
                label.append(0)  # low anxiety
            else:
                label.append(1)  # high anxiety
        
        data_y = pd.DataFrame({"subject": subjects, "label": label})
        # data_y = pd.DataFrame({"ranking": ranking_col})

        # out = pd.concat([data_x, data_y.iloc[:, 1:]], axis=1)
        
        return data_x, data_y
        # return out


class Train_WESAD:

    def get_labels():
        stai_scores = dr_w.get_stai_scores()
        stai_scores = stai_scores[stai_scores.iloc[:, 0] != 3.0].reset_index(drop=True)  # remove subject 3 due to NaNs in Medi_2 phase
        dim_scores_valence = dr_w.get_dim_scores(dim_type="valence")
        dim_scores_valence = dim_scores_valence[dim_scores_valence.iloc[:, 0] != 3.0].reset_index(drop=True)
        dim_scores_arousal = dr_w.get_dim_scores(dim_type="arousal")
        dim_scores_arousal = dim_scores_arousal[dim_scores_arousal.iloc[:, 0] != 3.0].reset_index(drop=True)

        return stai_scores, dim_scores_arousal, dim_scores_valence

    def get_wesad_data(metrics, phases, verbose=False, label_type="stai", normalize=True, threshold="fixed"):
        """
        label_type: "stai", "arousal", "valence", "all"
            label_type == "all": classification between stress and non-stress phases
        """
        metrics_folder = dr_w.Paths.METRICS
        stai_scores, dim_scores_arousal, dim_scores_valence = Train_WESAD.get_labels()

        columns = metrics.copy()
        columns.insert(0, "subject")
        
        data_x = []
        data_y = []

        for phase in phases:
            if verbose: print(f"Generating features for phase {phase} " + "-"*30)
            phase_id = phases.index(phase)
            features = []

            for i in range(len(metrics)):
                metric = metrics[i]
                if verbose: print(f"Generating features for metric {metric}")
                file = os.path.join(metrics_folder, f"{metric}_{phase}.csv")
                arr = pd.read_csv(file, index_col=[0])
                arr = arr[arr.iloc[:, 0] != 3.0].reset_index(drop=True).to_numpy()  # remove subject 3 due to NaNs in Medi_2 phase

                if i == 0:  # subject IDs
                    # ids = arr.iloc[:, 0]
                    ids = arr[:, 0]
                    ids = pd.DataFrame(data=ids, columns=["subject"])
                    features.append(ids)
                col_mean = np.nanmean(arr, axis=1)
                idx = np.where(np.isnan(arr))
                arr[idx] = np.take(col_mean, idx[0])
                arr = np.nan_to_num(arr)
                arr = np.mean(arr[:, 1:], axis=1)
                arr = np.reshape(arr, (arr.size, 1))
                arr = pd.DataFrame(data=arr, columns=[f"{metric}"])
                features.append(arr)

            x = pd.concat(features, axis=1)
            if label_type == "all":
                for scores in [stai_scores, dim_scores_arousal, dim_scores_valence]:
                    x = pd.concat([x, scores.iloc[:, 1:]], axis=1)
            phase = pd.DataFrame(data=[phase_id for _ in range(x.shape[0])])
            x.insert(1, "phaseId", phase)
                
            data_x.append(x)
        
        data_x = pd.concat(data_x).reset_index(drop=True)

        subjects = data_x.loc[:, "subject"]
        phase_col = data_x.loc[:, "phaseId"]

        if label_type == "all":
            y_labels = []
            for i in range(phase_col.shape[0]):
                if phase_col.iloc[i] == 1:  # TSST phase
                    y_labels.append(1)
                else:
                    y_labels.append(0)
            data_y = pd.Series(data=y_labels)
            data_x = data_x.drop("phaseId", axis=1)
        else:
            if label_type == "stai":
                scores = stai_scores
                if threshold == "fixed":
                    label_mean = 50
            elif label_type == "valence":
                scores = dim_scores_valence
            elif label_type == "arousal":
                scores = dim_scores_arousal
            else:
                raise ValueError(f"Invalid label type: {label_type}")

            columns = scores.columns

            y_labels = []
            for i in range(scores.shape[0]):
                if threshold != "fixed":
                    label_mean = scores.iloc[i, 1:].mean()
                labels = [scores.iloc[i, 0]]  # subject ID
                for j in range(1, scores.shape[1]):
                    if scores.iloc[i, j] < label_mean:
                        labels.append(0)
                    else:
                        labels.append(1)
                y_labels.append(labels)
            y_labels = pd.DataFrame(data=y_labels, columns=columns)

            for i in range(data_x.shape[0]):
                s = subjects.iloc[i]
                p = int(phase_col.iloc[i])
                label = y_labels.loc[y_labels["subject"] == s].iloc[0, p+1]
                data_y.append(label)
        data_y = pd.DataFrame({"subject": subjects, "label": data_y})

        if normalize:
            for metric in metrics:
                data_col = data_x[metric]
                data_col = (data_col - data_col.min())/(data_col.max() - data_col.min())
                data_x[metric] = data_col

        return data_x, data_y

    
class Train_POPANE:

    def get_popane_data(study, metrics, phases, verbose=False, normalize=True, label_type="affect", threshold="fixed"):
        metrics_folder = os.path.join(dr_p.Paths.METRICS, study)
        columns = metrics.copy()
        columns.insert(0, "subject")
        
        columns = metrics.copy()
        columns.insert(0, "subject")
        
        data_x = []
        data_y = []

        for phase in phases:
            if verbose: print(f"Generating features for phase {phase} " + "-"*30)
            phase_id = phases.index(phase)
            features = []

            for i in range(len(metrics)):
                metric = metrics[i]
                if verbose: print(f"Generating features for metric {metric}")
                file = os.path.join(metrics_folder, f"{metric}_{phase}.csv")
                arr = pd.read_csv(file, index_col=[0])
                arr = arr[arr.iloc[:, 0] != 3.0].reset_index(drop=True).to_numpy()  # remove subject 3 due to NaNs in Medi_2 phase

                if i == 0:  # subject IDs
                    # ids = arr.iloc[:, 0]
                    ids = arr[:, 0]
                    ids = pd.DataFrame(data=ids, columns=["subject"])
                    features.append(ids)
                col_mean = np.nanmean(arr, axis=1)
                idx = np.where(np.isnan(arr))
                arr[idx] = np.take(col_mean, idx[0])
                arr = np.nan_to_num(arr)
                arr = np.mean(arr[:, 1:], axis=1)
                arr = np.reshape(arr, (arr.size, 1))
                arr = pd.DataFrame(data=arr, columns=[f"{metric}"])
                # if arr.isnull().values.any():
                #     print(arr)
                features.append(arr)
            # for arr in features:
            #     print(arr.shape)
            x = pd.concat(features, axis=1)
            x.insert(1, "phaseId", phase_id)
            # if x.isnull().values.any():
            #     print(x)
            data_x.append(x)

        data_x = pd.concat(data_x).reset_index(drop=True)

        subjects = data_x.loc[:, "subject"]
        y_labels = []
        # TODO: need to edit phase labels
        if label_type == "arousal":
            for i in range(data_x.shape[0]):
                if phases[data_x.loc[i, "phaseId"]] in dr_p.HIGH_AROUSAL:
                    y_labels.append(0)
                else:
                    y_labels.append(1)
        elif label_type == "valence":
            for i in range(data_x.shape[0]):
                if phases[data_x.loc[i, "phaseId"]] in dr_p.HIGH_AROUSAL:
                    y_labels.append(0)
                else:
                    y_labels.append(1)
        else:  # label_type == affect
            self_report_df = pd.read_csv(os.path.join(dr_p.Paths.METRICS, study, "self_reports.csv"), index_col=0)
            for i in range(data_x.shape[0]):
                row = self_report_df.loc[self_report_df["subject"] == 1, :].iloc[:, 1:].replace(-1, np.NaN)
                phase = phases[data_x.loc[i, "phaseId"]]
                if threshold == "fixed":
                    mean_report = 5
                else:
                    mean_report = np.nanmean(row)
                if row.loc[:, phase][0] < mean_report:
                    y_labels.append(0)
                else:
                    y_labels.append(1)

        y_labels = pd.Series(data=y_labels)
        data_y = pd.DataFrame({"subject": subjects, "label": y_labels})

        if normalize: 
            for metric in metrics:
                data_col = data_x[metric]
                data_col = (data_col - data_col.min())/(data_col.max() - data_col.min())
                data_x[metric] = data_col

        return data_x, data_y


class Train_SFI:

    def get_sfi_data(metrics, phases, verbose=False, normalize=True):
        metrics_folder = dr_s.Paths.METRICS
        columns = metrics.copy()
        columns.insert(0, "subject")
        
        data_x = []
        data_y = []

        features = []
        for i in range(len(metrics)):
            metric = metrics[i]
            if verbose: print(f"Generating features for metric {metric}")
            file = os.path.join(metrics_folder, f"{metric}.csv")
            arr = pd.read_csv(file, index_col=[0])

            if i == 0:  # subject IDs
                # ids = arr.iloc[:, 0]
                ids = arr.iloc[:, 0]
                ids = pd.DataFrame(data=ids, columns=["subject"])
                features.append(ids)
            col_mean = np.nanmean(arr, axis=0)
            idx = np.where(np.isnan(arr))
            if idx[0].size > 0 and idx[1].size > 0:
                arr.iloc[idx] = np.take(col_mean, idx[1])
            arr = np.nan_to_num(arr)
            arr = np.mean(arr[:, 1:], axis=1)
            arr = np.reshape(arr, (arr.size, 1))
            arr = pd.DataFrame(data=arr, columns=[f"{metric}"])
            features.append(arr)

        x = pd.concat(features, axis=1)
            
        data_x.append(x)
        
        data_x = pd.concat(data_x).reset_index(drop=True)

        subjects = data_x.loc[:, "subject"]

        y_labels = []
        for phase in phases:
            if phase == "BIOFEEDBACK-REST":
                y_labels.append(0)
            else:
                y_labels.append(1)
        y_labels = pd.Series(data=y_labels)

        data_y = pd.DataFrame({"subject": subjects, "label": y_labels})

        for metric in metrics:
            data_col = data_x[metric]
            data_col = (data_col - data_col.min())/(data_col.max() - data_col.min())
            data_x[metric] = data_col

        return data_x, data_y


class Train_Multi_Dataset:
    
    def train_across_datasets(models, dataset_a_x, dataset_a_y, dataset_b_x, dataset_b_y, test_size=0.80, by_subject=True, save_metrics=True, target_names=["A", "B"], get_shap_values=False):
        """
        test_size: Proportion of dataset_b to hold out for model testing.
        """
        out = {}
        x_train_a, y_train_a, x_test_a, y_test_a, test_subjects = train_test_split(dataset_a_x, dataset_a_y, test_size=0.0, by_subject=by_subject)
        x_train_b, y_train_b, x_test_b, y_test_b, test_subjects = train_test_split(dataset_b_x, dataset_b_y, test_size=test_size, by_subject=by_subject)
        # print(f"x_train: {x_train.shape}")
        # print(f"y_train: {y_train.shape}")
        x_train = pd.concat([x_train_a, x_train_b])
        y_train = pd.concat([y_train_a, y_train_b])
        x_test = x_test_b
        y_test = y_test_b.loc[:, "label"]
        
        # print("Training data: ")
        # print(y_train.loc[:, "label"].value_counts())
        # print("Testing data: ")
        # print(y_test.loc[:].value_counts())

        for model_name in models.keys():
            model = models[model_name]
            model = model.fit(x_train, y_train.loc[:, "label"])
            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            if save_metrics:
                precision = precision_score(y_test, y_pred, zero_division=1)
                recall = recall_score(y_test, y_pred, zero_division=1)
                f1 = f1_score(y_test, y_pred, zero_division=1)
                auc = roc_auc_score(y_test, y_pred)
                report = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "auc": auc,
                    "actual vs pred": [y_test, y_pred]
                }
            else:
                report = None
            if get_shap_values and acc > 0.65:
                try: 
                    explainer = shap.Explainer(model)
                except Exception as e:
                    explainer = shap.Explainer(model.predict, x_train)
                shap_values = explainer(x_test)
            else:
                shap_values = None
            out[model_name] = (acc, report, shap_values)
        return out



if __name__ == "__main__":
    pass
