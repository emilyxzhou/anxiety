import datetime
import glob
import importlib
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import random
import shap
import scipy.signal as ss

import tools.data_reader_apd as dr_a
import tools.data_reader_ascertain as dr_asc
import tools.data_reader_popane as dr_p
import tools.data_reader_sfi as dr_s
import tools.data_reader_wesad as dr_w
import tools.preprocessing as preprocessing

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from scipy.fft import fft, fftfreq, fftshift
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, precision_score, f1_score, \
    recall_score, roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import normalize

import cvxopt.solvers
cvxopt.solvers.options['show_progress'] = False

import warnings
warnings.filterwarnings(
    "ignore", 
    category=RuntimeWarning
)

random.seed(datetime.datetime.now().timestamp())


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


def kfold_train_test_split(x, y, test_size=0.1, is_resample=False, folds=1):
    n_splits = int(1/test_size)
    cv_index = random.choice(range(0, n_splits))
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    subjects = list(x.loc[:, "subject"])
    for i, (train_index, test_index) in enumerate(sgkf.split(x, y.loc[:, "label"], subjects)):
        if i == cv_index:
            break
    x_train = x.iloc[train_index, :].reset_index(drop=True)
    y_train = y.iloc[train_index, :].reset_index(drop=True)
    x_test = x.iloc[test_index, :].reset_index(drop=True)
    y_test = y.iloc[test_index, :].reset_index(drop=True)
    subjects = list(x_train.loc[:, "subject"])

    sgkf = StratifiedGroupKFold(n_splits=folds)
    return sgkf.split(x_train, y_train.loc[:, "label"], subjects), x_train, y_train, x_test, y_test


def resample(x, y, threshold=0.333):
    if not x.empty and not y.empty:
        _, counts = np.unique(y["label"], return_counts=True)
        if counts.shape[0] > 1:
            neg = counts[0]
            pos = counts[1]
            try:
                if neg / pos < threshold:
                    print(f"Ratio of negative to positive labels ({neg/pos}) is under {threshold}, oversampling negative class.")
                    oversample = SMOTE()
                    # oversample = RandomOverSampler(sampling_strategy=threshold)
                    x, y = oversample.fit_resample(x, y["label"])
                    y = pd.concat([x["subject"], y], axis=1)
                elif pos / neg < threshold:
                    print(f"Ratio of positive to negative labels ({pos/neg}) is under {threshold}, oversampling positive class.")
                    oversample = SMOTE()
                    # oversample = RandomOverSampler(sampling_strategy=threshold)
                    x, y = oversample.fit_resample(x, y["label"])
                    y = pd.concat([x["subject"], y], axis=1)
            except Exception as e:
                print("Error in resampling train/test data")
    return x, y


def grid_search_cv(
        models, parameters, x, y, test_size=0.1, 
        by_subject=True, save_metrics=True, get_importance=False, print_preds=False, 
        is_resample=False, drop_subject=True, folds=1
    ):
    """
    return: model_data = {{
        model_name: {
            "cv": ___,
            "best_params": ___,
            "best_model": ___,
            "train": ___,
            "test": ___,
        }
    }}
    """
    cv, x_train, y_train, x_test, y_test = kfold_train_test_split(x, y, test_size, is_resample=is_resample, folds=folds)
    # subjects = np.unique(list(x_train.loc[:, "subject"]))
    # subjects.sort()
    # print(f"x_train subjects: {subjects}")
    # subjects = np.unique(list(x_test.loc[:, "subject"]))
    # subjects.sort()
    # print(f"x_test subjects: {subjects}")
    if drop_subject:
        x_train = x_train.drop("subject", axis=1)
        y_train = y_train.drop("subject", axis=1).to_numpy().flatten()
        x_test = x_test.drop("subject", axis=1)
        y_test = y_test.drop("subject", axis=1).to_numpy().flatten()
    cv_list = []

    # HYPERPARAMETER GRID SEARCH
    for _, (train_index, test_index) in enumerate(cv):
        cv_list.append((train_index, test_index))
    model_data["cv"] = cv_list
    model_data = {name: {} for name in models.keys()}
    for model_name in models.keys():
        model = models[model_name]
        params = parameters[model_name]
        clf = GridSearchCV(model, params, cv=cv_list, scoring="roc_auc")
        clf.fit(x_train, y_train)
        best_params = clf.best_params_
        best_model = clf.best_estimator_
        model_data[model_name]["best_params"] = best_params
        model_data[model_name]["best_model"] = best_model
    model_data["train"] = (x_train, y_train)
    model_data["test"] = (x_test, y_test)

    return model_data


def feature_selection(models, cv, x_train, y_train, n_features=5):
    model_data = {name: {} for name in models.keys()}
    for model_name in models.keys():
        model = models[model_name]
        sfs = SequentialFeatureSelector(model, n_features_to_select=n_features)


def test_model(models, x_test, y_test):
    model_data = {name: {} for name in models.keys()}
    for model_name in models.keys():
        model = models[model_name]
        if model_name == "random":
            y_pred = [random.choice([0, 1]) for i in range(x_test.shape[0])]
        else:
            if model_name == "LogReg":
                y_pred = model.predict_proba(x_test)
                y_pred = (y_pred[:, 1] >= 0.7).astype(int)
            else:
                y_pred = model.predict(x_test)
        unique, counts = np.unique(y_test, return_counts=True)
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        print(f"Model {model_name}, Actual: {unique}, {counts}, Predictions: {unique_pred}, {counts_pred}")
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)
            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
            # Uncomment the following two lines to see ROC plot
            # display.plot()
            # plt.show()
            auc_score = roc_auc_score(y_test, y_pred)
        except Exception as e:
            print(e)
            print("Only one class present in y_true. ROC AUC score is not defined in that case. Setting AUC score to -1.")
            auc_score = -1
        report = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc_score
        }

        if model_name == "XGB":
            importance = model.feature_importances_
        elif model_name == "DT":
            importance = model.feature_importances_
        elif model_name == "LogReg":
            importance = model.coef_[0]
        elif model_name == "RF":
            importance = model.feature_importances_
        elif model_name == "SVM":
            importance = model.coef_[0]
        else:
            print(f"Feature importance not available for {model_name}")
            importance = None

        model_data[model_name]["performance"] = (acc, report, importance)

    return model_data


class Train_ASCERTAIN:

    def get_ascertain_data(metrics, verbose=False, label_type="Arousal", threshold="dynamic", normalize=True, binary_labels=True, combine_phases=False):
        metrics_folder = dr_asc.Paths.METRICS
        
        columns = metrics.copy()
        columns.insert(0, "subject")
        
        data_x = []
        data_y = []

        for c, clip in enumerate(dr_asc.CLIPS):
            if verbose: print(f"Generating features for clip {clip} " + "-"*30)
            phase_id = dr_asc.CLIPS.index(clip)
            features = []

            for i in range(len(metrics)):
                metric = metrics[i]
                if verbose: print(f"Generating features for metric {metric}")
                file = os.path.join(metrics_folder, f"{metric}_Clip{clip}.csv")
                if combine_phases:
                    arr = pd.read_csv(file, index_col=[0])
                    arr[arr == -np.inf] = np.nan
                    if i == 0:  # subject IDs
                        ids = arr.iloc[:, 0].tolist()
                        ids = pd.DataFrame(data=ids, columns=["subject"])
                        features.append(ids)
                    arr = np.nanmean(arr.iloc[:, 1:], axis=1)
                    arr = np.reshape(arr, (arr.size, 1))
                    arr = pd.DataFrame(data=arr, columns=[f"{metric}"])
                    features.append(arr)
                else:
                    arr = pd.read_csv(file, index_col=[0]).reset_index(drop=True).to_numpy()
                    arr[arr == -np.inf] = 0
                    num_cols = arr.shape[1]
                    count = 0
                    for row in range(arr.shape[0]):  # split each 50-second segment into a separate sample
                        for col in range(num_cols-1):
                            if i == 0:  # first metric
                                features.append([arr[row, 0], phase_id, arr[row][col+1]])  # subject, phase ID, value
                            else:
                                if count < len(features):
                                    features[count].append(arr[row, col+1])
                            count += 1

            if combine_phases:
                x = pd.concat(features, axis=1)
                phase = pd.DataFrame(data=[phase_id for _ in range(x.shape[0])])
                x.insert(1, "phaseId", phase)
            else:
                x = pd.DataFrame(features, columns=["subject", "phaseId"] + metrics)
            data_x.append(x)
        
        data_x = pd.concat(data_x).reset_index(drop=True)
        if "lf_rr" in metrics and "hf_rr" in metrics:
            data_x["lf_hf_ratio"] = data_x["lf_rr"] / data_x["hf_rr"]
            metrics.append("lf_hf_ratio")

        subjects = data_x.loc[:, "subject"]
        phase_col = data_x.loc[:, "phaseId"]

        if label_type == dr_asc.SelfReports.AROUSAL:
            scores = dr_asc.get_self_reports(label_type)
        elif label_type == dr_asc.SelfReports.VALENCE:
            scores = dr_asc.get_self_reports(label_type)
        else:
            raise ValueError(f"Invalid label type: {label_type}")

        columns = scores.columns
        
        y_labels = []
        for i in range(scores.shape[0]):
            if binary_labels:
                if threshold != "fixed":
                    s = scores.iloc[i, 0]
                    label_means = dr_asc.get_mean_self_reports(label_type)
                    label_mean = label_means[label_means["subject"] == s].loc[:, "mean"].iloc[0]
                else:
                    if label_type == dr_asc.SelfReports.AROUSAL:
                        label_mean = 4
                    if label_type == dr_asc.SelfReports.VALENCE:
                        label_mean = 0
                labels = [scores.iloc[i, 0]]  # subject ID
                for j in range(1, scores.shape[1]):
                    if scores.iloc[i, j] < label_mean:
                        labels.append(0)
                    else:
                        labels.append(1)
            else:
                labels = [scores.iloc[i, 0]]  # subject ID
                for j in range(1, scores.shape[1]):
                    labels.append(scores.iloc[i, j])
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
        if "lf_hf_ratio" in metrics:
            metrics.remove("lf_hf_ratio")
        return data_x, data_y


class Train_APD:
    
    def get_ratings(phases, threshold="dynamic"):
        """ Returns SUDS labels, ranges from 20-80. """
        label_dict = {
            "Baseline_Rest": "Baseline_SUDS", 
            "BugBox_Relax": "BugBox_Relax_SUDS",
            "BugBox_Anticipate": "BugBox_Preparation_SUDS",
            "BugBox_Exposure": "BugBox_Exposure_SUDS", 
            "BugBox_Break": "BugBox_Break_SUDS", 
            "Speech_Relax": "Speech_Relax_SUDS",
            "Speech_Anticipate": "Speech_SUDS",
            "Speech_Exposure": "Speech_Exposure_SUDS",
            "Speech_Break": "Speech_Break_SUDS"
        }

        ha_participant_indices = [
            'P4', 'P6', 'P7', 'P8', 'P10', 'P12', 'P15', 'P16', 'P18', 'P22', 'P26', 'P27', 'P29', 'P31', 'P32', 'P33', 'P35', 'P42', 'P45', 'P47', 'P48', 'P49', 'P54', 'P55', 'P66', 'P69'
        ]

        la_participant_indices = [
            'P14', 'P21', 'P23', 'P25', 'P34', 'P39', 'P43', 'P46', 'P51', 'P57', 'P71', 'P72', 'P77', 'P78', 'P79', 'P80', 'P82', 'P83', 'P84', 'P85', 'P87', 'P88', 'P89', 'P91', 'P92', 'P93'
        ]

        participant_file = os.path.join(dr_a.Paths.DATA_DIR, "participants_details.csv")
        df = pd.read_csv(participant_file)

        suds_labels = ["Participant"] + [label_dict[phase] for phase in phases]
        suds_df = df[suds_labels]
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

        if threshold == "fixed": 
            ha_suds_df['mean'] = [30 for _ in range(ha_suds_df.shape[0])]
            la_suds_df['mean'] = [30 for _ in range(la_suds_df.shape[0])]
        else:
            ha_suds_df['mean'] = ha_suds_df.iloc[:, 1:].mean(axis=1)
            la_suds_df['mean'] = la_suds_df.iloc[:, 1:].mean(axis=1)

        ha_rankings = ha_suds_df.rename(columns={c: suds_labels.index(c)-1 for c in ha_suds_df.columns[1:-1]}).reset_index(drop=True)
        la_rankings = la_suds_df.rename(columns={c: suds_labels.index(c)-1 for c in la_suds_df.columns[1:-1]}).reset_index(drop=True)

        return ha_rankings, la_rankings


    def get_apd_data_ranking(metrics, phases, verbose=False, anxiety_label_type=None, threshold="dynamic", normalize=True, binary_labels=True, combine_phases=False):
        """
        anxiety_label_type: can be None, "Trait", "Anxiety", "Depression", "Gender", "Random"
            - Adds an extra feature vector 
            - Labels generated based on SUDS responses
        """
        metrics_folder = dr_a.Paths.METRICS
        ha_rankings, la_rankings = Train_APD.get_ratings(phases, threshold)

        columns = metrics.copy()
        columns.insert(0, "subject")

        data_x = []
        data_y = pd.concat([ha_rankings, la_rankings], axis=0).reset_index(drop=True)

        for phase in phases:
            phase_id = phases.index(phase)
            if verbose: print(f"Generating features for phase {phase}, ID {phase_id} " + "-"*30)
            ha_features = []
            la_features = []

            for i in range(len(metrics)):
                metric = metrics[i]
                if verbose: print(f"Generating features for metric {metric}")
                file = os.path.join(metrics_folder, f"{metric}_{phase}_ha.csv")
                arr = pd.read_csv(file, index_col=[0]).to_numpy()
                col_mean = np.nanmean(arr, axis=1)
                idx = np.where(np.isnan(arr))
                arr[idx] = np.take(col_mean, idx[0])
                arr = np.nan_to_num(arr)
                if combine_phases:
                    if i == 0:  # subject IDs
                        ids = np.reshape(arr[:, 0], (arr[:, 0].size, 1))
                        ids = pd.DataFrame(data=ids, columns=["subject"])
                        ha_features.append(ids)
                    arr = np.mean(arr[:, 1:], axis=1)
                    arr = np.reshape(arr, (arr.size, 1))
                    arr = pd.DataFrame(data=arr, columns=[f"{metric}"])
                    ha_features.append(arr)
                else:
                    num_cols = arr.shape[1]
                    count = 0
                    for row in range(arr.shape[0]):  # split each 50-second segment into a separate sample
                        for col in range(num_cols-1):
                            if i == 0:  # first metric
                                ha_features.append([arr[row, 0], phase_id, arr[row][col+1]])  # subject, phase ID, value
                            else:
                                ha_features[count].append(arr[row, col+1])
                            count += 1

                file = os.path.join(metrics_folder, f"{metric}_{phase}_la.csv")
                arr = pd.read_csv(file, index_col=[0]).to_numpy()
                col_mean = np.nanmean(arr, axis=1)
                idx = np.where(np.isnan(arr))
                arr[idx] = np.take(col_mean, idx[0])
                arr = np.nan_to_num(arr)
                if combine_phases:
                    if i == 0:  # subject IDs
                        ids = np.reshape(arr[:, 0], (arr[:, 0].size, 1))
                        ids = pd.DataFrame(data=ids, columns=["subject"])
                        la_features.append(ids)
                    arr = np.mean(arr[:, 1:], axis=1)
                    arr = np.reshape(arr, (arr.size, 1))
                    arr = pd.DataFrame(data=arr, columns=[f"{metric}"])
                    la_features.append(arr)
                else:
                    num_cols = arr.shape[1]
                    count = 0
                    for row in range(arr.shape[0]):  # split each 50-second segment into a separate sample
                        for col in range(num_cols-1):
                            if i == 0:  # first metric
                                la_features.append([arr[row, 0], phase_id, arr[row][col+1]])  # subject, phase ID, value
                            else:
                                la_features[count].append(arr[row, col+1])
                            count += 1

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
            if combine_phases:
                ha_features = pd.concat(ha_features, axis=1)
                la_features = pd.concat(la_features, axis=1)
            else:
                ha_features = pd.DataFrame(ha_features, columns=["subject", "phaseId"] + metrics)
                la_features = pd.DataFrame(la_features, columns=["subject", "phaseId"] + metrics)
            x = pd.concat([ha_features, la_features], axis=0)
            if combine_phases:
                phase = pd.DataFrame(data=[phase_id for _ in range(x.shape[0])])
                x.insert(1, "phaseId", phase)

            if anxiety_label_type is not None: 
                x.insert(1, "anxietyGroup", anxiety_label)

            data_x.append(x)
        
        try:
            data_x = pd.concat(data_x).reset_index(drop=True)
            data_x["lf_hf_ratio"] = data_x["lf_rr"] / data_x["hf_rr"]
            metrics.append("lf_hf_ratio")
        except Exception:
            pass
            # print("Error in generating lf_hf_ratio")

        if normalize:
            # normalize columns
            for metric in metrics:
                data_col = data_x[metric]
                data_col = (data_col - data_col.min())/(data_col.max() - data_col.min())
                data_x[metric] = data_col
            # normalize rows
            # for i in range(data_x.shape[0]):
            #     data_row = data_x.loc[data_x.index[i], metrics]
            #     data_row = (data_row - data_row.min())/(data_row.max() - data_row.min())
            #     data_x.loc[data_x.index[i], metrics] = data_row

        subjects = data_x.loc[:, "subject"]
        phase_col = data_x.loc[:, "phaseId"]
        label = []
        for i in range(data_x.shape[0]):
            s = int(subjects.iloc[i])
            p = int(phase_col.iloc[i])
            rating = data_y.loc[data_y["subject"] == s].loc[:, p].values[0]
            mean = data_y.loc[data_y["subject"] == s].loc[:, 'mean'].values[0]  # 'mean' = fixed threshold if that's the option used
            if rating < mean:
                label.append(0)  # low anxiety
            else:
                label.append(1)  # high anxiety
        data_y = pd.DataFrame({"subject": subjects, "label": label})

        if "lf_hf_ratio" in metrics:
            metrics.remove("lf_hf_ratio")
        return data_x, data_y
        # return out


class Train_WESAD:

    def get_labels(phases):
        stai_scores = dr_w.get_stai_scores(phases)
        # stai_scores = stai_scores[stai_scores.iloc[:, 0] != 3.0].reset_index(drop=True)  # remove subject 3 due to NaNs in Medi_2 phase
        dim_scores_valence = dr_w.get_dim_scores(phases, dim_type="valence")
        # dim_scores_valence = dim_scores_valence[dim_scores_valence.iloc[:, 0] != 3.0].reset_index(drop=True)
        dim_scores_arousal = dr_w.get_dim_scores(phases, dim_type="arousal")
        # dim_scores_arousal = dim_scores_arousal[dim_scores_arousal.iloc[:, 0] != 3.0].reset_index(drop=True)

        return stai_scores, dim_scores_arousal, dim_scores_valence

    def get_wesad_data(metrics, phases, verbose=False, label_type="stai", normalize=True, threshold="dynamic", binary_labels=True, combine_phases=False):
        """
        label_type: "stai", "arousal", "valence", "all"
            label_type == "all": classification between stress and non-stress phases
        """
        metrics_folder = dr_w.Paths.METRICS
        stai_scores, dim_scores_arousal, dim_scores_valence = Train_WESAD.get_labels(phases)

        columns = metrics.copy()
        columns.insert(0, "subject")
        
        data_x = []
        data_y = [] 

        for p, phase in enumerate(phases):
            # print(phase)
            if verbose: print(f"Generating features for phase {phase} " + "-"*30)
            phase_id = phases.index(phase)
            features = []
            for i in range(len(metrics)):
                metric = metrics[i]
                # print(f"----- {metric}")
                if verbose: print(f"Generating features for metric {metric}")
                file = os.path.join(metrics_folder, f"{metric}_{phase}.csv")
                arr = pd.read_csv(file, index_col=[0]).reset_index(drop=True).to_numpy()
                # arr = arr[arr.iloc[:, 0] != 3.0].reset_index(drop=True).to_numpy()  # remove subject 3 due to NaNs in Medi_2 phase

                col_mean = np.nanmean(arr, axis=1)
                idx = np.where(np.isnan(arr))
                arr[idx] = np.take(col_mean, idx[0])
                arr = np.nan_to_num(arr)

                if combine_phases:
                    if i == 0:  # subject IDs
                        # ids = arr.iloc[:, 0]
                        ids = arr[:, 0]
                        ids = pd.DataFrame(data=ids, columns=["subject"])
                        features.append(ids)
                    arr = np.mean(arr[:, 1:], axis=1) 
                    arr = np.reshape(arr, (arr.size, 1))
                    arr = pd.DataFrame(data=arr, columns=[f"{metric}"])
                    features.append(arr)
                else:
                    num_cols = arr.shape[1]
                    count = 0
                    for row in range(arr.shape[0]):  # split each 50-second segment into a separate sample
                        for col in range(num_cols-1):
                            if i == 0:  # first metric
                                features.append([arr[row, 0], phase_id, arr[row][col+1]])  # subject, phase ID, value
                            else:
                                features[count].append(arr[row, col+1])
                            count += 1

            if combine_phases:
                x = pd.concat(features, axis=1)
                if label_type == "all":
                    for scores in [stai_scores, dim_scores_arousal, dim_scores_valence]:
                        x = pd.concat([x, scores.iloc[:, 1:]], axis=1)
                phase = pd.DataFrame(data=[phase_id for _ in range(x.shape[0])])
                x.insert(1, "phaseId", phase)
            else:
                x = pd.DataFrame(features, columns=["subject", "phaseId"] + metrics)
            
            data_x.append(x)
        
        data_x = pd.concat(data_x).reset_index(drop=True)
        if "lf_rr" in metrics and "hf_rr" in metrics:
            data_x["lf_hf_ratio"] = data_x["lf_rr"] / data_x["hf_rr"]
            metrics.append("lf_hf_ratio")

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
                    label_mean = 60
            elif label_type == "valence":
                scores = dim_scores_valence
                # not sure what the fixed threshold is for the DIM survey
            elif label_type == "arousal":
                scores = dim_scores_arousal
            else:
                raise ValueError(f"Invalid label type: {label_type}")

            columns = scores.columns

            y_labels = []
            for i in range(scores.shape[0]):
                if binary_labels:
                    if threshold != "fixed":
                        label_mean = scores.iloc[i, 1:].mean()
                    labels = [scores.iloc[i, 0]]  # subject ID
                    for j in range(1, scores.shape[1]):
                        if scores.iloc[i, j] < label_mean:
                            labels.append(0)
                        else:
                            labels.append(1)
                else:
                    labels = [scores.iloc[i, 0]]  # subject ID
                    for j in range(1, scores.shape[1]):
                        labels.append(scores.iloc[i, j])
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
        if "lf_hf_ratio" in metrics:
            metrics.remove("lf_hf_ratio")
        return data_x, data_y

    
class Train_POPANE:

    def get_popane_data(study, metrics, phases, verbose=False, normalize=True, label_type="affect", threshold="dynamic", binary_labels=True):
        metrics_folder = os.path.join(dr_p.Paths.METRICS, study)
        columns = metrics.copy()
        columns.insert(0, "subject")
        
        columns = metrics.copy()
        columns.insert(0, "subject")
        
        data_x = []
        data_y = []

        for phase in phases:
            phase_id = phases.index(phase)
            if verbose: print(f"Generating features for phase {phase}, ID {phase_id} " + "-"*30)
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
        data_x = data_x[data_x["subject"].notna()].reset_index(drop=True)

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
                if phases[data_x.loc[i, "phaseId"]] in dr_p.HIGH_VALENCE:
                    y_labels.append(0)
                else:
                    y_labels.append(1)
        else:  # label_type == affect
            self_report_df = pd.read_csv(os.path.join(dr_p.Paths.METRICS, study, "self_reports.csv"), index_col=0)
            for i in range(data_x.shape[0]):
                subject = data_x.iloc[i, :].loc["subject"]
                row = self_report_df.loc[self_report_df["subject"] == subject, :].iloc[:, 1:].replace(-1, np.NaN)
                phase = phases[data_x.loc[i, "phaseId"]]
                if binary_labels:
                    if threshold == "fixed":
                        mean_report = 5
                    else:
                        mean_report = np.nanmean(row)
                    try:
                        if row.loc[:, phase].iloc[0] < mean_report:
                            y_labels.append(0)
                        else:
                            y_labels.append(1)
                    except Exception as e:
                        continue
                else:
                    print(row.loc[:, phase].iloc[0])
                    y_labels.append(row.loc[:, phase].iloc[0])
        
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
    
    def train_across_datasets(models, dataset_a_x, dataset_a_y, dataset_b_x, dataset_b_y, test_size=0.80, by_subject=True, save_metrics=True, target_names=["1", "0"], get_importance=False, is_resample=False, drop_subject=False):
        """
        test_size: Proportion of dataset_b to hold out for model testing.
        """
        out = {}
        x_train_a, y_train_a, x_test_a, y_test_a = kfold_train_test_split(dataset_a_x, dataset_a_y, test_size=0.0, by_subject=by_subject, is_resample=is_resample, folds=1)
        x_train_b, y_train_b, x_test_b, y_test_b = kfold_train_test_split(dataset_b_x, dataset_b_y, test_size=test_size, by_subject=by_subject, is_resample=is_resample, folds=1)
        x_train = pd.concat([x_train_a[0], x_train_b[0]])
        y_train = pd.concat([y_train_a[0], y_train_b[0]])
        x_test = x_test_b[0]

        # print("A -----")
        # print(x_train_a[0].shape)
        # print(y_train_a[0].shape)
        # print(x_test_a[0].shape)
        # print(y_test_a[0].shape)
        
        # print("B -----")
        # print(x_train_b[0].shape)
        # print(y_train_b[0].shape)
        # print(x_test_b[0].shape)
        # print(y_test_b[0].shape)

        if drop_subject:
            x_train = x_train.drop("subject", axis=1)
            x_test = x_test.drop("subject", axis=1)
        y_test = y_test_b[0].loc[:, "label"]

        print(f"y_train:\n{y_train.loc[:, 'label'].value_counts()}")
        print(f"y_test:\n{y_test.value_counts()}")

        for model_name in models.keys():
            if model_name == "random":
                y_pred = [random.choice([0, 1]) for i in range(x_test.shape[0])]
            else:
                model = models[model_name]
                model = model.fit(x_train, y_train.loc[:, "label"])
                y_pred = model.predict(x_test)

            unique, counts = np.unique(y_pred, return_counts=True)
            print(f"Model {model_name}, Predictions: {unique}, {counts}")

            acc = accuracy_score(y_test, y_pred)
            if save_metrics:
                precision = precision_score(y_test, y_pred, zero_division=1)
                recall = recall_score(y_test, y_pred, zero_division=1)
                f1 = f1_score(y_test, y_pred, zero_division=1)
                try:
                    auc = roc_auc_score(y_test, y_pred)
                except Exception as e:
                    print("Only one class present in y_true. ROC AUC score is not defined in that case. Setting AUC score to -1.")
                    auc = -1
                report = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "auc": auc,
                    "actual vs pred": [y_test, y_pred]
                }
            else:
                report = None
            if get_importance:
                # print("Calculating shap values")
                if model_name == "XGB":
                    explainer  = shap.Explainer(model, x_train)
                    # importance = explainer(x_train, check_additivity=False)
                    importance = model.feature_importances_
                elif model_name == "LogReg":
                    importance = model.coef_[0]
                elif model_name == "DT":
                    importance = model.feature_importances_
                elif model_name == "RF":
                    importance = model.feature_importances_
                elif model_name == "SVM":
                    importance = model.coef_[0]
                else:
                    print(f"Feature importance not available for {model_name}")
                    importance = None
            else:
                importance = None
            out[model_name] = (acc, report, importance)
        return out


if __name__ == "__main__":
    pass
