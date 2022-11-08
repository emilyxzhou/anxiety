# IMPORTING MODULES
import os
import sys
# NOTE: THIS IS THE CORRECT WAY TO DO RELATIVE IMPORTS
cvx_eda_path = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'cvxEDA', 'src'))
src_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(cvx_eda_path)
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
    HA = [df.iloc[1:, 1:].to_numpy().astype(np.float32) for df in HA]
    LA = [df.iloc[1:, 1:].to_numpy().astype(np.float32) for df in LA]

    if convert_sr:
        HA = [samplerate.resample(HA[i], ratio=100.0 / 250.0) for i in range(len(HA))]
        LA = [samplerate.resample(LA[i], ratio=100.0 / 250.0) for i in range(len(LA))]
    
    if data_type == dr.DataTypes.ECG:
        if is_clean_ecg:
            HA = [clean_ecg(HA[i]) for i in range(len(HA))]
            LA = [clean_ecg(LA[i]) for i in range(len(LA))]

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
    """Perform FFT on a single 1D time signal, returns a frequencies and magnitudes"""
    data_amp = np.abs(fft(data.flatten()))
    data_amp = data_amp
    n = data_amp.size
    data_amp = data_amp / n
    freq = fftfreq(n, d=1/fs)
    # freq = fftshift(freq)  # don't need this
    freq = np.reshape(freq, (freq.size, 1))
    data_amp = np.reshape(data_amp, (data_amp.size, 1))
    return freq, data_amp


def moving_average(x, w):
    x = x.flatten()
    return np.convolve(x, np.ones(w), "valid") / w


def clean_ecg(ecg_signal):
    if ecg_signal.size <= 1:
        # print("ECG signal has size 0, returning None")
        return None
    fs = FS_DICT[dr.DataTypes.ECG]
    ecg_signal = hp.scale_data(ecg_signal)
    ecg_signal = hp.remove_baseline_wander(ecg_signal.flatten(), fs)
    ecg_signal = hp.enhance_peaks(ecg_signal, iterations=2)
    
    sos = ss.butter(N=2, Wn=0.667, btype="highpass", fs=fs, output="sos")
    filtered = ss.sosfilt(sos, ecg_signal)
    # filtered = moving_average(filtered, 8)
    filtered = np.reshape(filtered, (filtered.size, 1))
    filtered = clean_RR(filtered)
    return filtered


def clean_RR(clean_ecg_signal):
    fs = FS_DICT[dr.DataTypes.ECG]
    sos_low = ss.butter(N=3, Wn=[0.75], btype="low", fs=fs, output="sos")
    sos_high = ss.butter(N=3, Wn=[50], btype="low", fs=fs, output="sos")
    sos_bs = ss.butter(N=3, Wn=[2.5, 8], btype="bandstop", fs=fs, output="sos")
    filtered = ss.sosfilt(sos_low, clean_ecg_signal)
    filtered = ss.sosfilt(sos_high, filtered)
    filtered = ss.sosfilt(sos_bs, filtered)
    return filtered


def get_hf_rr(ecg):
    fs = FS_DICT[dr.DataTypes.ECG]
    n = ecg.size
    if n == 0:
        print("ECG signal has length 0, returning None")
        return None
    start = 0
    window_size = int(55*fs)
    stop = start + window_size
    out = []
    while stop < n:
        stop = start + window_size
        segment = ecg[start:stop]
        freq, amp = calculate_fft_1d(segment, fs)
        
        low = 0.15
        high = 0.4
        freq[freq < low] = 0
        freq[freq > high] = 0
        
        power = np.multiply(freq, amp).sum()
        out.append(power)

        start += int(5*fs)
    return np.asarray(out)


def get_lf_rr(ecg):
    fs = FS_DICT[dr.DataTypes.ECG]
    n = ecg.size
    if n == 0:
        print("ECG signal has length 0, returning None")
        return None
    start = 0
    window_size = int(55*fs)
    stop = start + window_size
    out = []
    while stop < n:
        stop = start + window_size
        segment = ecg[start:stop]
        freq, amp = calculate_fft_1d(segment, fs)
        
        low = 0.04
        high = 0.15
        freq[freq < low] = 0
        freq[freq > high] = 0
        
        power = np.multiply(freq, amp).sum()
        out.append(power)

        start += int(5*fs)
    return np.asarray(out)


def get_SC_metrics(eda):
    """Returns r (phasic), t (tonic) components of the input EDA signal."""
    if eda.size == 0:
        print("Warning: EDA signal has size 0, retuning None")
        return None, None
    fs = FS_DICT[dr.DataTypes.EDA]
    eda = eda.astype(np.double)
    # sos = ss.butter(N=3, Wn=4.0, btype="lowpass", fs=fs, output="sos")
    # filtered = ss.sosfilt(sos, eda_signal)
    # sr = 200*(272+filtered)/(752-filtered)
    # sc = 1/sr
    [r, p, t, l, d, e, obj] = cvxEDA(eda, 1./fs, options={"show_progress": False})
    r = np.log10(r + 1)
    p = np.log10(p + 1)
    return r, p


def get_mean_SCL(eda_signal):
    fs = FS_DICT[dr.DataTypes.EDA]
    _, scl = get_SC_metrics(eda_signal)
    if scl is None:
        return None
    n = scl.size
    start = 0
    window_size = int(60*fs)
    stop = start + window_size
    out = []
    while stop < n:
        stop = start + window_size
        segment = scl[start:stop]
        segment_mean = np.mean(segment)
        out.append(segment_mean)
        start += int(5*fs)
    return np.asarray(out)


def get_SCR_rate(eda_signal):
    fs = FS_DICT[dr.DataTypes.EDA]
    scr, _ = get_SC_metrics(eda_signal)
    if scr is None:
        return None

    grad = np.gradient(scr)
    n = grad.size
    start = 0
    window_size = int(60*fs)
    stop = start + window_size
    # threshold = max()

    out = []
    while stop < n:
        stop = start + window_size
        # segment = grad[start:stop]
        segment = scr[start:stop]
        # num_peaks = len(list(itertools.groupby(segment, lambda x: x > 0)))
        peaks, _ = ss.find_peaks(segment)
        num_peaks = len(peaks)
        out.append(num_peaks//2 + 1)
        start += int(5*fs)
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
