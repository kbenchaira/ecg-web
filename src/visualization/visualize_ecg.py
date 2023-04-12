import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


def data_visualization(ecg_data, FS):
    ecg_data = ecg_data.reshape(2700, 6)
    columns = 2
    rows = 3
    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(15, 10), sharex=True)

    # Set the x-axis values (time)
    time = np.arange(0, 2700)

    # Plot each channel of the ECG data in a separate subplot
    for i, ax in np.ndenumerate(axes):
        index = i[0] * columns + i[1]
        ax.plot(time, ecg_data[:, index], label=f'Segment {index + 1}')
        ax.set_title(f'Segment {index + 1}', fontsize=14)
        ax.set_xlabel('Samples', fontsize=8)
        ax.set_ylabel('mV', fontsize=8)
        ax.set_ylim([ecg_data[:, index].min() - 0.5, ecg_data[:, index].max() + 0.5])
        ax.legend(loc='upper right')
        ax.grid()

    # Adjust the layout and display the plot
    fig.tight_layout()
    return fig


def plot_ecg(uploaded_ecg, FS):
    ecg_1d = uploaded_ecg.reshape(-1)
    N = len(ecg_1d)
    time = np.arange(N) / FS
    p = FS * 5

    num_subplots = int(np.ceil(len(time) / p))
    ncols = 2
    nrows = int(np.ceil(num_subplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 10))

    for i, ax in enumerate(axes.flatten()):
        segment = ecg_1d[i * p:(i * p + p)]
        time_segment = time[i * p:(i * p + p)]
        ax.plot(time_segment, segment)
        ax.set_title(f'Segment from {i * 5} to {5 * i + 5} seconds', fontsize=7)
        ax.set_xlabel('Time in s', fontsize=5)
        ax.set_ylabel('ECG in mV', fontsize=5)
        ax.set_ylim([segment.min() - 0.5, segment.max() + 0.5])
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))

        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))

        ax.grid(which='major', color='#CCCCCC', linestyle='--')
        ax.grid(which='minor', color='#CCCCCC', linestyle=':')

        ax.xaxis.set_tick_params(labelsize=6)
        ax.yaxis.set_tick_params(labelsize=6)

    fig.tight_layout()
    return fig