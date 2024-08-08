import numpy as np
from feature.chord import *
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import welch

class ChordProgressionAnalyzer():
    def __init__(self, chord_progressions):
        self.chord_progressions = chord_progressions
        self.vectorized_progressions = self._vectorize_progressions()

    def _vectorize_progressions(self):
        return np.array([
            extractChordNumeralValues(progression)
            for progression in self.chord_progressions
        ])

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

        fig, axs = plt.subplots(n_rows, 1, figsize=(15, 3 * n_rows))
        fig.suptitle('Signal Variance Analysis', fontsize=20)

        # Plot original signal
        axs[0].plot(signal, color='blue')
        axs[0].set_title('Original Signal')
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