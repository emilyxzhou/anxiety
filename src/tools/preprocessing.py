# IMPORTING MODULES
import os
import sys
# NOTE: THIS IS THE CORRECT WAY TO DO RELATIVE IMPORTS
cvx_eda_path = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'cvxEDA', 'src'))
pyeda_path = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'pyEDA'))
pyeda_main_path = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'pyEDA', 'main'))
src_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(cvx_eda_path)
sys.path.append(pyeda_path)
sys.path.append(pyeda_main_path)
sys.path.append(src_path)
import glob
import heartpy as hp
import itertools
import numpy as np
import pandas as pd
import samplerate
import scipy
import scipy.signal as ss

import tools.data_reader_apd as dr

from cvxEDA import cvxEDA
from scipy.fft import fft, fftfreq

import biosppy

import pyhrv
import pyhrv.time_domain as td


FS_DICT = {
    dr.DataTypes.ANKLE_L: 50.0,
    dr.DataTypes.ANKLE_R: 50.0,
    dr.DataTypes.WRIST_L: 50.0,
    dr.DataTypes.WRIST_R: 50.0,
    dr.DataTypes.EDA: 50.0,
    dr.DataTypes.ECG: 250.0,
    dr.DataTypes.POSTURE: 1.0
}

DATA_TYPE_DIMENSIONS = {
    dr.DataTypes.ANKLE_L: 9,
    dr.DataTypes.ANKLE_R: 9,
    dr.DataTypes.WRIST_L: 9,
    dr.DataTypes.WRIST_R: 9,
    dr.DataTypes.EDA: 1,
    dr.DataTypes.ECG: 1,
    dr.DataTypes.POSTURE: 2
}

DATA_TYPE_DIMENSION_LABELS = {
    dr.DataTypes.ANKLE_L: ["a_x", "a_y", "a_z", "w_x", "w_y", "w_z", "roll", "yaw", "pitch"],
    dr.DataTypes.ANKLE_R: ["a_x", "a_y", "a_z", "w_x", "w_y", "w_z", "roll", "yaw", "pitch"],
    dr.DataTypes.WRIST_L: ["a_x", "a_y", "a_z", "w_x", "w_y", "w_z", "roll", "yaw", "pitch"],
    dr.DataTypes.WRIST_R: ["a_x", "a_y", "a_z", "w_x", "w_y", "w_z", "roll", "yaw", "pitch"],
    dr.DataTypes.EDA: ["Sensor reading"],
    dr.DataTypes.ECG: ["ECG reading"],
    dr.DataTypes.POSTURE: ["Degrees", "Activity level (VMU)"]
}


def load_data(task, data_type, phase, convert_sr=False, is_clean_ecg=True):
    """Returns a list of np arrays without timestamps and headings"""
    HA = dr.get_dataframes_from_files(
            glob.glob(dr.Paths.PARTICIPANT_DATA_DIR + "\\" + dr.Groups.HA + f"\\*\\{task}\\{data_type}_{phase}.csv")
        )
    LA = dr.get_dataframes_from_files(
        glob.glob(dr.Paths.PARTICIPANT_DATA_DIR + "\\" + dr.Groups.LA + f"\\*\\{task}\\{data_type}_{phase}.csv")
    )
    # get rid of timestamp and heading
    HA = [df.drop(df.columns[[1]], axis=1).iloc[1:, :].astype(np.float32) for df in HA]
    LA = [df.drop(df.columns[[1]], axis=1).iloc[1:, :].astype(np.float32) for df in LA]

    if convert_sr:
        for i in range(len(HA)):
            new_col = samplerate.resample(HA[i].iloc[:, -1], ratio=100.0 / 250.0)
            HA[i] = HA[i].iloc[:, :-1]
            HA[i] = pd.concat([HA[i], new_col], axis=1)
        for i in range(len(LA)):
            new_col = samplerate.resample(LA[i].iloc[:, -1], ratio=100.0 / 250.0)
            LA[i] = LA[i].iloc[:, :-1]
            LA[i] = pd.concat([LA[i], new_col], axis=1)
    
    if data_type == dr.DataTypes.ECG:
        for i in range(len(HA)):
            if is_clean_ecg:
                new_col = clean_ecg(HA[i].iloc[:, -1]).reset_index(drop=True)
                num_rows = new_col.shape[0]
                HA[i] = HA[i].iloc[:num_rows, :-1].reset_index(drop=True)
                HA[i] = pd.concat([HA[i], new_col], axis=1)
        for i in range(len(LA)):
            if is_clean_ecg:
                new_col = clean_ecg(LA[i].iloc[:, -1]).reset_index(drop=True)
                num_rows = new_col.shape[0]
                LA[i] = LA[i].iloc[:num_rows, :-1].reset_index(drop=True)
                LA[i] = pd.concat([LA[i], new_col], axis=1)
    return HA, LA


def calculate_group_metric(group, task, data_type, phase, metric):
    """
    :return: x = nd.array, out = np.ndarray with shape (num_elements, num_features) for metrics other than 'fft'
        If metric is 'fft', return shape is (num_elements, num_features, num_samples, 2) where the 4th dimension
        represents frequencies and corresponding amplitudes.
    """
    if group not in ["HA", "LA"]:
        raise ValueError(f"Invalid group type: {group}")

    ha, la = load_data(task, data_type, phase, convert_sr=False)

    if group == "HA": 
        data = ha
    else:
        data = la

    # data = pad_list_of_ndarrays(data)
    data = crop_list_of_ndarrays(data)
    metric_dim = data.ndim - 1
    feature_dim = 1

    if metric == "mean":
        out = np.mean(data, axis=metric_dim)
        x = np.arange(0, out.size, 1)
    elif metric == "median":
        out = np.median(data, axis=metric_dim)
        x = np.arange(0, out.size, 1)
    elif metric == "fft":
        x, ft_list = calculate_fft(data, FS_DICT[data_type], feature_dim=feature_dim)
        out = np.dstack(ft_list)
    elif metric == "fft mean":
        x, ft_list = calculate_fft(data, FS_DICT[data_type], feature_dim=feature_dim)
        out = np.dstack(ft_list)
        out = np.mean(out, axis=metric_dim)
    elif metric == "fft median":
        x, ft_list = calculate_fft(data, FS_DICT[data_type], feature_dim=feature_dim)
        out = np.dstack(ft_list)
        out = np.median(out, axis=metric_dim)
    else:
        raise ValueError(f"Invalid metric: {metric}")

    return x, out


def calculate_fft(data, fs, feature_dim):
    """Returns the FFT frequencies and amplitudes"""
    out = []
    num_features = data.shape[feature_dim]
    num_samples = data.shape[-1]
    sample_dim = data.ndim-1
    if num_features == 1:
        for i in range(num_samples):
            input = np.take(data, indices=[i], axis=sample_dim)
            freq, amp = calculate_fft_1d(input, fs)
            out.append(amp)
    else:
        for i in range(num_samples):
            sample = np.take(data, indices=[i], axis=sample_dim)
            # print(f"sample: {sample.shape}")
            for j in range(num_features):
                input = np.take(sample, indices=[j], axis=feature_dim)
                # print(f"input: {input.shape}")
                freq, amp = calculate_fft_1d(input, fs)
                # print(f"amp: {amp.shape}")
                if j == 0:
                    fft = amp
                else:
                    fft = np.hstack([fft, amp])
            # fft = np.transpose(fft)
            out.append(fft)
    return freq, out
    

def calculate_fft_1d(data, fs):
    """Perform FFT on a single 1D time signal, returns frequencies and magnitudes"""
    data = np.asarray(data)
    data_amp = np.abs(fft(data.flatten()))
    data_amp = data_amp
    n = data_amp.size
    # data_amp = data_amp / n
    data_amp = data_amp[0:n//2]
    freq = fftfreq(n, d=1/fs)[0:n//2]
    # freq = fftshift(freq)  # don't need this
    freq = np.reshape(freq, (freq.size, 1))
    data_amp = np.reshape(data_amp, (data_amp.size, 1))
    return freq, data_amp


def moving_average(x, w):
    x = x.flatten()
    return np.convolve(x, np.ones(w), "valid") / w


def clean_ecg(ecg_signal, fs):
    if ecg_signal.size <= 1:
        print("ECG signal has size 0, returning zero DataFrame")
        return pd.DataFrame([0.0])
    ecg_signal = hp.filter_signal(ecg_signal, cutoff=50.0, sample_rate=fs, filtertype="lowpass")
    # ecg_signal = hp.remove_baseline_wander(ecg_signal, fs)
    ecg_signal = hp.scale_data(ecg_signal)
    # print(f"Size of scaled ECG signal: {ecg_signal.shape}")
    # ecg_signal = hp.enhance_ecg_peaks(
    #     ecg_signal, fs, aggregation='median', iterations=5
    # )
    # print(f"Size of enhanced ECG signal: {ecg_signal.shape}")
    ecg_signal = pd.DataFrame(ecg_signal)
    
    # print("Stats --------------------")
    # print(f"Min: {ecg_signal.min().iloc[0]}")
    # print(f"Max: {ecg_signal.max().iloc[0]}")
    # print(f"Median: {ecg_signal.median().iloc[0]}")
    # print(f"Mean: {ecg_signal.mean().iloc[0]}")
    
    # sos = ss.butter(N=2, Wn=0.667, btype="highpass", fs=fs, output="sos")
    # ecg_signal = ss.sosfilt(sos, ecg_signal)
    # ecg_signal = moving_average(ecg_signal, 8)
    # ecg_signal = np.reshape(ecg_signal, (ecg_signal.size, 1))
    # ecg_signal = clean_RR(ecg_signal)
    # ecg_signal = pd.DataFrame(ecg_signal.flatten())
    return ecg_signal


def clean_RR(clean_ecg_signal):
    fs = FS_DICT[dr.DataTypes.ECG]
    sos_low = ss.butter(N=3, Wn=[0.75], btype="low", fs=fs, output="sos")
    sos_high = ss.butter(N=3, Wn=[50], btype="low", fs=fs, output="sos")
    sos_bs = ss.butter(N=3, Wn=[2.5, 8], btype="bandstop", fs=fs, output="sos")
    filtered = ss.sosfilt(sos_low, clean_ecg_signal)
    filtered = ss.sosfilt(sos_high, filtered)
    filtered = ss.sosfilt(sos_bs, filtered)
    return filtered


def get_hf_rr(ecg, fs=FS_DICT[dr.DataTypes.ECG], window_size=50):
    n = ecg.size
    if n == 0:
        print("ECG signal has length 0, returning None")
        return None
    start = 0
    window_size = int(window_size*fs)
    stop = start + window_size
    out = []
    if stop > n:
        segment = ecg
        freq, amp = calculate_fft_1d(segment, fs)
        
        low = 0.15
        high = 0.4
        freq[freq < low] = 0
        freq[freq > high] = 0
        amp = np.multiply(freq, amp)
        
        power = np.multiply(amp, amp).sum() # Parseval's theorem
        out.append(power)
    else:
        while stop < n:
            stop = start + window_size
            try:
                segment = ecg.iloc[start:stop]
            except AttributeError:
                segment = ecg[start:stop]

            freq, amp = calculate_fft_1d(segment, fs)
            
            low = 0.15
            high = 0.4
            freq[freq < low] = 0
            freq[freq > high] = 0
            amp = np.multiply(freq, amp)
            
            power = np.multiply(amp, amp).sum() # Parseval's theorem
            out.append(power)

            start = stop
    return np.asarray(out)


def get_lf_rr(ecg, fs=FS_DICT[dr.DataTypes.ECG], window_size=50):
    n = ecg.size
    if n == 0:
        print("ECG signal has length 0, returning None")
        return None
    start = 0
    window_size = int(window_size*fs)
    stop = start + window_size
    out = []
    if stop > n:
        segment = ecg
        freq, amp = calculate_fft_1d(segment, fs)
        
        low = 0.04
        high = 0.15
        freq[freq < low] = 0
        freq[freq > high] = 0
        amp = np.multiply(freq, amp)
        
        power = np.multiply(amp, amp).sum() # Parseval's theorem
        out.append(power)
    else:
        while stop < n:
            stop = start + window_size
            try:
                segment = ecg.iloc[start:stop]
            except AttributeError:
                segment = ecg[start:stop]
            freq, amp = calculate_fft_1d(segment, fs)
            
            low = 0.04
            high = 0.15
            freq[freq < low] = 0
            freq[freq > high] = 0
            amp = np.multiply(freq, amp)
            
            power = np.multiply(amp, amp).sum() # Parseval's theorem
            out.append(power)

            start = stop
    return np.asarray(out)


def get_ecg_metrics_pyhrv(ecg_signal, fs=FS_DICT[dr.DataTypes.ECG], window_size=50):
    n = ecg_signal.size
    if n == 0:
        print("ECG signal has length 0, returning None")
        return None
    
    metrics_dict = {
        "rmssd": [],
        # "ibi": [],
        "sdnn": []
    }

    start = 0
    window_size = int(window_size*fs)
    stop = start + window_size
    if stop > n:
        segment = ecg_signal
        t, filtered_signal, rpeaks = biosppy.signals.ecg.ecg(signal=segment, sampling_rate=fs, show=False)[:3]
        rmssd = td.rmssd(rpeaks=t[rpeaks])["rmssd"]
        # ibi = td.ibi(rpeaks=rr_ints)
        sdnn = td.sdnn(rpeaks=t[rpeaks])["sdnn"]

        metrics_dict["rmssd"].append(rmssd)
        # metrics_dict["ibi"].append(ibi)
        metrics_dict["sdnn"].append(sdnn)
    else:
        while stop < n:
            stop = start + window_size
            try:
                segment = ecg_signal.iloc[start:stop]
            except AttributeError:
                segment = ecg_signal[start:stop]
            try:
                t, filtered_signal, rpeaks = biosppy.signals.ecg.ecg(signal=segment, sampling_rate=fs, show=False)[:3]
                rmssd = td.rmssd(rpeaks=t[rpeaks])["rmssd"]
                # ibi = td.ibi(rpeaks=rr_ints)
                sdnn = td.sdnn(rpeaks=t[rpeaks])["sdnn"]
            except Exception as e:
                print(e)
                rmssd = np.nan
                # ibi = np.nan
                sdnn = np.nan

            metrics_dict["rmssd"].append(rmssd)
            # metrics_dict["ibi"].append(ibi)
            metrics_dict["sdnn"].append(sdnn)

            start = stop
    
    return metrics_dict


def get_bpm_biosppy(ecg_signal, fs=FS_DICT[dr.DataTypes.ECG], window_size=50):
    n = ecg_signal.size
    if n == 0:
        print("ECG signal has length 0, returning None")
        return None
    
    start = 0
    window_size = int(window_size*fs)
    stop = start + window_size
    out = []
    if stop > n:
        segment = ecg_signal
        data = biosppy.signals.ecg.ecg(signal=segment, sampling_rate=fs, show=False)
        bpm = data["heart_rate"].tolist()
        out.append(np.mean(bpm))

    else:
        while stop < n:
            stop = start + window_size
            try:
                segment = ecg_signal.iloc[start:stop]
            except AttributeError:
                segment = ecg_signal[start:stop]
            try:
                data = biosppy.signals.ecg.ecg(signal=segment, sampling_rate=fs, show=False)
                bpm = data["heart_rate"].tolist()
                out.append(np.mean(bpm))
            except Exception as e:
                bpm = [np.nan]
                out.append(bpm)

            start = stop
    
    return out


def get_SC_metrics(eda, fs=FS_DICT[dr.DataTypes.EDA]):
    """Returns r (phasic), t (tonic) components of the input EDA signal."""
    if eda.size == 0:
        print("Warning: EDA signal has size 0, retuning None")
        return None, None
    eda = eda.astype(np.double)
    # sos = ss.butter(N=3, Wn=4.0, btype="lowpass", fs=fs, output="sos")
    # filtered = ss.sosfilt(sos, eda_signal)
    # sr = 200*(272+filtered)/(752-filtered)
    # sc = 1/sr
    [r, p, t, l, d, e, obj] = cvxEDA(eda, 1./fs, options={"show_progress": False})
    r = np.log10(r + 1)
    t = np.log10(t + 1)
    # print(r)
    # print(t)
    return r, t
    # return phasic, tonic


def get_mean_SCL(eda_signal, fs=FS_DICT[dr.DataTypes.EDA], window_size=50):
    _, scl = get_SC_metrics(eda_signal, fs)
    if scl is None:
        print("mean SCL is None")
        return None
    n = scl.size
    start = 0
    window_size = int(window_size*fs)
    stop = start + window_size
    out = []
    if n < stop:
        segment = scl
        segment_mean = np.mean(segment)
        print(f"mean SCL: {segment_mean}")
        out.append(segment_mean)
    while stop < n:
        stop = start + window_size
        segment = scl[start:stop]
        segment_mean = np.mean(segment)
        print(f"mean SCL: {segment_mean}")
        out.append(segment_mean)
        start = stop
    return np.asarray(out)


def  get_SCR_rate(eda_signal, fs=FS_DICT[dr.DataTypes.EDA], window_size=50):
    scr, _ = get_SC_metrics(eda_signal, fs)
    if scr is None:
        print("SCR rate is None")
        return None

    grad = np.gradient(scr)
    n = grad.size
    start = 0
    window_size = int(window_size*fs)
    stop = start + window_size
    # threshold = max()
    out = []
    if n < stop:
        segment = scr
        peaks, _ = ss.find_peaks(segment)
        num_peaks = len(peaks)
        print(f"SCR rate: {num_peaks}")
        out.append(num_peaks//2 + 1)
    while stop < n:
        stop = start + window_size
        # segment = grad[start:stop]
        segment = scr[start:stop]
        # num_peaks = len(list(itertools.groupby(segment, lambda x: x > 0)))
        peaks, _ = ss.find_peaks(segment)
        num_peaks = len(peaks)
        print(f"SCR rate: {num_peaks}")
        out.append(num_peaks//2 + 1)
        start = stop
    return np.asarray(out)


def clean_acc_data(acc_signal):
    if acc_signal.size <= 1:
        print("ACC signal has size 0, returning zero DataFrame")
        return pd.DataFrame([0.0])
    fs = FS_DICT[dr.DataTypes.ANKLE_L]
    sos = ss.butter(N=2, Wn=0.8, btype="highpass", fs=fs, output="sos")
    acc_signal = ss.sosfilt(sos, acc_signal)
    return acc_signal


def get_peak_acc_value(acc_signal, acc_type):
    """
    Calculate peak acceleration in the x-y plane.
    """
    if acc_type == "torso":
        fs = FS_DICT[dr.DataTypes.POSTURE]
    elif acc_type == "wrist":
        fs = FS_DICT[dr.DataTypes.WRIST_L]
    elif acc_type == "ankle":
        fs = FS_DICT[dr.DataTypes.ANKLE_L]

    n = acc_signal.shape[0]
    start = 0
    window_size = int(window_size*fs)
    stop = start + window_size

    out = []
    while stop < n:
        stop = start + window_size
        segment = acc_signal.iloc[start:stop, 0:2]
        segment = np.square(segment)
        segment = segment.sum(axis=1)
        segment = np.sqrt(segment)
        peak = segment.max()
        out.append(peak)
        start = stop
    return np.asarray(out)


def get_mean_ankle_activity(acc_signal):
    fs = FS_DICT[dr.DataTypes.ANKLE_L]

    n = acc_signal.shape[0]
    start = 0
    window_size = int(60*fs)
    stop = start + window_size

    out = []
    while stop < n:
        stop = start + window_size
        segment = acc_signal.iloc[start:stop, 0:3]
        segment = np.square(segment)
        segment = np.mean(segment, axis=0)
        segment = np.sqrt(segment)
        mean = np.mean(segment)
        out.append(mean)
        start = stop
    return np.asarray(out)


def get_mean_posture(posture_signal):
    fs = FS_DICT[dr.DataTypes.POSTURE]

    n = posture_signal.shape[0]
    start = 0
    window_size = int(60*fs)
    stop = start + window_size

    out = []
    while stop < n:
        stop = start + window_size
        segment = posture_signal.iloc[start:stop, 0]
        mean = np.mean(segment)
        out.append(mean)
        start = stop
    return np.asarray(out)


def get_mean_activity_torso(posture_signal):
    fs = FS_DICT[dr.DataTypes.POSTURE]

    n = posture_signal.shape[0]
    start = 0
    window_size = int(60*fs)
    stop = start + window_size

    out = []
    while stop < n:
        stop = start + window_size
        segment = posture_signal.iloc[start:stop, 1]
        mean = np.mean(segment)
        out.append(mean)
        start = stop
    return np.asarray(out)


def pad_list_of_ndarrays(array_list):
    """
    Pads a list of np.ndarrays to the max size in each dimension.
    Each ndarray must have the same number of dimensions.
    :return: 3D np.ndarray with shape (num_elements, num_features, num_samples)
    """
    # print(f"ORIGINAL SHAPES: {[arr.shape for arr in array_list]}")
    n_dims = array_list[0].ndim
    max_dim_sizes = []
    for i in range(n_dims):
        max_size = np.max([arr.shape[i] for arr in array_list])
        max_dim_sizes.append(max_size)
    for i in range(len(array_list)):
        arr = array_list[i].copy()
        pad_sizes = []
        for j in range(len(max_dim_sizes)):
            diff = max_dim_sizes[j] - array_list[i].shape[j]
            pad_sizes.append((0, diff))
        pad_sizes = tuple(pad_sizes)
        arr = np.pad(arr, pad_sizes, mode="constant")
        array_list[i] = arr

    # print(f"FINAL SHAPES: {[arr.shape for arr in array_list]}")
    out = np.dstack(array_list)
    return out


def crop_list_of_ndarrays(array_list):
    """
    Crops a list of np.ndarrays to the min size in each dimension.
    Each ndarray must have the same number of dimensions.
    :return: 3D np.ndarray with shape (num_elements, num_features, num_samples)
    """
    # print(f"ORIGINAL SHAPES: {[arr.shape for arr in array_list]}")
    i = len(array_list) - 1
    while i >= 0:
        if array_list[i].size == 0:
            print(f"No data values for participant at index {i}; removing from list.")
            array_list.pop(i)
        i -= 1

    min_num_samples = np.min([arr.shape[0] for arr in array_list])

    for i in range(len(array_list)):
        indices = np.arange(0, min_num_samples, 1)
        arr = np.take(array_list[i], indices, 0).astype(np.float32)
        array_list[i] = arr

    # print(f"FINAL SHAPES: {[arr.shape for arr in array_list]}")
    out = np.dstack(array_list)
    return out


if __name__ == "__main__":
    convert_sr = False
    task = dr.Tasks.SPEAKING
    data_type = dr.DataTypes.ECG
    phase = dr.Phases.SPEECH_EXPOSURE

    fs = FS_DICT[data_type]
    n_dim = DATA_TYPE_DIMENSIONS[data_type]

    # x, ha_ecg_mean = calculate_group_metric("HA", task, data_type, phase, metric="mean")
    # x, la_ecg_mean = calculate_group_metric("LA", task, data_type, phase, metric="mean")
    # x, ha_ecg_med = calculate_group_metric("HA", task, data_type, phase, metric="median")
    # x, la_ecg_med = calculate_group_metric("LA", task, data_type, phase, metric="median")
    # x, ha_ecg_fft = calculate_group_metric("HA", task, data_type, phase, metric="fft")
    # x, la_ecg_fft = calculate_group_metric("LA", task, data_type, phase, metric="fft")
