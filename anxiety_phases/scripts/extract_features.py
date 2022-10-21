import glob
import heartpy as hp
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from skimage.restoration import denoise_wavelet
from tools import data_reader_apd as dr
from tools import display_tools as dt


def extract_baseline_rest_ecg_features(index):
    files = glob.glob(
        os.path.join(dr.Paths.PARTICIPANT_DATA_DIR, "*", f"p_{index}", "baseline", "Heart_Baseline_Rest.csv")
    )
    if len(files) == 0:
        print(f"No heart data found for participant {index}.")
        return []
    df = pd.read_csv(files[0])
    ecg_data = df["ECG reading"].to_numpy()
    ecg_data = denoise_wavelet(
        ecg_data, method="BayesShrink", mode="soft", wavelet_levels=3, wavelet="sym8",
        rescale_sigma="True"
    )

    ecg_data = hp.scale_data(ecg_data)
    d, m = hp.process(ecg_data, 250.0)
    print(d.keys())
    hp.plotter(d, m, title=f"Scaled ECG data, participant {index}")
    plt.show()

    ecg_data_enhanced_peaks = hp.enhance_peaks(ecg_data)
    d, m = hp.process(ecg_data_enhanced_peaks, 250.0)
    hp.plotter(d, m, title=f"Scaled ECG data, enhanced peaks, participant {index}")
    plt.show()

    ecg_data_enhanced_ecg_peaks = hp.enhance_ecg_peaks(ecg_data, 250.0)
    d, m = hp.process(ecg_data_enhanced_ecg_peaks, 250.0)
    hp.plotter(d, m, title=f"Scaled ECG data, enhanced ECG peaks, participant {index}")
    plt.show()


if __name__ == "__main__":
    extract_baseline_rest_ecg_features(4)
