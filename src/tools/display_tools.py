import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools import preprocessing

colors = ['#D62728', '#2C9F2C', '#FD7F23', '#1F77B4', '#9467BD',
          '#8C564A', '#7F7F7F', '#1FBECF', '#E377C2', '#BCBD27']


# labels = {0: "LA", 1: "HA"}

def plot_1d_data(data, data_type=None, title=None, labels=None, start=0, stop=None):
    plt.figure(figsize=(17, 7))

    if stop is None:
        stop = data.size

    x = np.arange(start, stop, 1)
    y = data[start:stop].flatten()

    # print(f"Min amplitude: {np.min(y)}")
    # print(f"Max amplitude: {np.max(y)}")

    plt.plot(x, y, label=labels, color=colors[0], linewidth=2)
    plt.xlabel("Timestep")
    plt.ylabel(f"{data_type}, {labels}")
    plt.legend(loc="upper right")
    plt.title(title)

    y_min = np.min(y)
    y_max = np.max(y)
    y_med = np.median([y_min, y_max])
    plt.ylim(y_min - (y_med - y_min) / 5, y_max + (y_max - y_med) / 5)
    plt.tight_layout()


def plot_fft_data(freq, amp, data_type=None, title=None, labels=None, amp_threshold=None, freq_range=None):
    plt.figure(figsize=(11, 7))
    x = np.absolute(freq)
    y = np.abs(amp)
    max_index = np.argmax(y)
    # print(f"Value of max index {max_index}: {x[max_index]}, {y[max_index]}")

    if freq_range is None:
        freq_range = [0, np.max(x)]

    # print(f"Min frequency: {np.min(x)}")
    # print(f"Max frequency: {np.max(x)}")
    # print(f"Min amplitude: {np.min(y)}")
    # print(f"Max amplitude: {np.max(y)}")
    plt.stem(x, y, label=labels)
    plt.xlabel("Frequencies")
    plt.ylabel(data_type)
    plt.legend(loc="upper right")
    plt.title(title)

    y_min = np.min(y)
    y_max = np.max(y)
    y_med = np.median([y_min, y_max])
    plt.xlim(freq_range[0], freq_range[1])
    plt.ylim(0, y_max*1.1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pass
