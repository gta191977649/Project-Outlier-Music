import numpy as np
from feature.chord import *
from feature.pattern import *
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import welch
from matplotlib.gridspec import GridSpec
import japanize_matplotlib
class ChordProgressionAnalyzer():
    def __init__(self, chord_progressions):
        self.chord_progressions = chord_progressions
        self.vectorized_progressions = self._vectorize_progressions()

    def _vectorize_progressions(self):
        return np.array([
            extractChordNumeralValues(progression)
            for progression in self.chord_progressions
        ])
        # # Use TPS
        # return np.array([
        #     extractTontalPitchDistancePattern(progression,key="C:maj",mode="profile")
        #     for progression in self.chord_progressions
        # ])

    def plotHeatMap(self):
        sns.heatmap(self.vectorized_progressions)
        plt.tight_layout()

        plt.show()
    def plotDensity(self):
        sns.kdeplot(self.vectorized_progressions)
        plt.xticks(range(0, 13))  # Adjust the range as needed
        plt.tight_layout()
        plt.show()

    def plotConcatenatedSignal(self):
        signal = self.vectorized_progressions.flatten()
        plt.figure(figsize=(12, 3))
        plt.plot(signal,color="blue")
        plt.xlim(0, len(signal))
        plt.tight_layout()
        plt.show()

    def analyze_signal_variance(self, window_sizes=[4, 8, 16,24]):
        def rolling_variance(data, window):
            return np.array([np.var(data[max(0, i - window):i + 1]) for i in range(len(data))])

        signal = self.vectorized_progressions.flatten()

        # Calculate number of rows needed for subplots
        n_rows = len(window_sizes) + 2  # +2 for original signal and PSD

        fig, axs = plt.subplots(n_rows, 1, figsize=(10, 2 * n_rows))

        # Plot original signal
        axs[0].plot(signal, color='blue')
        axs[0].set_title('Chord Pattern Signal')
        axs[0].set_xlim(0, len(signal))

        # Plot rolling variances
        for i, window in enumerate(window_sizes, start=1):
            variance = rolling_variance(signal, window)
            axs[i].plot(variance, color='red')
            axs[i].set_title(f'Rolling Variance (Window Size: {window})')
            axs[i].set_xlim(0, len(signal))

        # Compute and plot power spectral density
        f, Pxx = welch(signal, nperseg=1024)
        axs[-1].semilogy(f, Pxx)
        axs[-1].set_xlim(0, max(f))
        axs[-1].set_title('Power Spectral Density')
        axs[-1].set_xlabel('Frequency')
        axs[-1].set_ylabel('Power/Frequency')

        plt.tight_layout()
        plt.show()

    def analyze_progression_position_compoment(self):
        x_transposed = self.vectorized_progressions.T
        num_rows = x_transposed.shape[0]
        num_columns = x_transposed.shape[1]
        x_range = 13  # x-axis range from 1 to 12

        fig = plt.figure(figsize=(10, 4 * num_rows))  # Increase the figure height to make room for larger heatmaps
        gs = GridSpec(num_rows * 2, 1, height_ratios=[4, 1] * num_rows)

        for i in range(num_rows):
            # Initialize an empty array for heatmap data
            heatmap_data = np.zeros((num_columns, x_range - 1))
            for j in range(num_columns):
                val = int(x_transposed[i, j])
                if 1 <= val < x_range:
                    heatmap_data[j, val - 1] = 1  # Mark the position with 1

            ax_heatmap = fig.add_subplot(gs[i * 2])
            sns.heatmap(heatmap_data, ax=ax_heatmap, cbar=False, cmap="binary_r", annot=False, linewidths=0,
                        linecolor='black')
            ax_heatmap.set_title(f'CHORD SEQUENCE POSITION: {i + 1}')
            ax_heatmap.set_ylabel('Song Index')
            ax_heatmap.set_xticks(np.arange(x_range - 1) + 0.5)
            ax_heatmap.set_xticklabels(np.arange(1, x_range))



            # Aligning the x-axis of the heatmap with the KDE plot
            ax_heatmap.set_xlim(0, x_range - 1)

            ax_kde = fig.add_subplot(gs[i * 2 + 1])
            sns.kdeplot(x_transposed[i, :], ax=ax_kde, color="black", bw_adjust=0.5, fill=True)
            ax_kde.set_xlim(1, 12)
            ax_kde.set_title(f'KDE: {i + 1}')
            ax_kde.set_ylabel('Density')
            ax_kde.set_xticks(np.arange(1, x_range))
            if i == num_rows - 1:
                ax_kde.set_xlabel('Value')

        plt.tight_layout()
        plt.show()