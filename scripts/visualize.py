# IMPORTING MODULES
import glob
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.signal as ss
import sys

import tools.data_reader as dr
import tools.display_tools as dt
import tools.preprocessing as preprocessing

from scipy.fft import fft, fftfreq, fftshift

# TEST

N = 500
fs = 500
n = np.linspace(0, 1/fs*N, N)
signal = 2*np.sin(n*2*np.pi*3) + np.sin(n*2*np.pi*15)
freq, amp = preprocessing.calculate_freq_data_1d(signal, fs=fs)

plt.plot(n, signal)
plt.show()

dt.plot_fft_data(freq, amp, freq_range=[0, 20])

fc = 10
w = fc / (fs / 2)

sos = ss.butter(N=2, Wn=5, btype="highpass", output="sos", fs=fs)
filtered = ss.sosfilt(sos, signal)
freq_f, amp_f = preprocessing.calculate_freq_data_1d(signal, fs=fs)

plt.figure()
plt.plot(n, filtered)
plt.show()

dt.plot_fft_data(freq_f, amp_f, freq_range=[0, 20])
