import numpy as np
from feature.chord import *
from feature.pattern import *
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import welch
from matplotlib.gridspec import GridSpec
import japanize_matplotlib
from sklearn.preprocessing import OneHotEncoder
import networkx as nx

class ChordProgressionAnalyzer:
    def __init__(self, chord_progressions):
        self.chord_progressions = chord_progressions
        self.vectorized_progressions = self._vectorize_progressions()

    def _vectorize_progressions(self, method="numeral"):
        if method == "numeral":
            return np.array([
                extractChordNumeralValues(progression)
                for progression in self.chord_progressions
            ])
        elif method == "onehot":
            flattened = np.array(self.chord_progressions).flatten().reshape(-1, 1)
            encoder = OneHotEncoder(sparse=False)
            onehot = encoder.fit_transform(flattened)
            return onehot.reshape(len(self.chord_progressions), -1)
        else:
            raise ValueError("Unsupported vectorization method")

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

    def analyze_progression_position_component(self):
        x_transposed = self.vectorized_progressions.T
        num_rows = x_transposed.shape[0]
        num_columns = x_transposed.shape[1]
        x_range = 13  # chord range from 1 to 12

        fig = plt.figure(figsize=(6, 4 * num_rows))  # Increased width for better readability
        gs = GridSpec(num_rows * 2, 1, height_ratios=[5, 2] * num_rows)

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
            ax_heatmap.set_ylabel('Pattern Index')

            # Set x-ticks to chord labels
            chord_labels = [number_to_chord_label(n) for n in range(1, x_range)]
            ax_heatmap.set_xticks(np.arange(x_range - 1) + 0.5)
            ax_heatmap.set_xticklabels(chord_labels, rotation=90, ha='right')

            # Aligning the x-axis of the heatmap with the KDE plot
            ax_heatmap.set_xlim(0, x_range - 1)

            ax_kde = fig.add_subplot(gs[i * 2 + 1])
            sns.kdeplot(x_transposed[i, :], ax=ax_kde, color="black", bw_adjust=0.5, fill=True)
            ax_kde.set_xlim(1, 12)
            ax_kde.set_title(f'KDE: {i + 1}')
            ax_kde.set_ylabel('Density')

            # Set x-ticks to chord labels for KDE plot
            ax_kde.set_xticks(np.arange(1, x_range))
            ax_kde.set_xticklabels(chord_labels, rotation=90, ha='right')

            if i == num_rows - 1:
                ax_kde.set_xlabel('Chord')

        plt.tight_layout()
        plt.show()

    def analyze_progression_position_component_kde_only(self):
        x_transposed = self.vectorized_progressions.T
        num_rows = min(x_transposed.shape[0], 4)  # Limit to 4 positions
        x_range = 13  # chord range from 1 to 12

        fig, axes = plt.subplots(num_rows, 1, figsize=(8, 1.5 * num_rows))
        if num_rows == 1:
            axes = [axes]  # Make axes iterable if there's only one subplot

        for i in range(num_rows):
            ax = axes[i]
            sns.kdeplot(x_transposed[i, :], ax=ax, color="black", bw_adjust=0.5, fill=True)

            ax.set_xlim(1, 12)
            ax.set_title(f'Position {i + 1}')
            ax.set_ylabel('Density')

            # Remove x-axis labels for all but the last subplot
            if i < num_rows - 1:
                ax.set_xticklabels([])
                ax.set_xlabel('')
            else:
                # Set x-ticks to chord labels only for the last subplot
                chord_labels = [number_to_chord_label(n) for n in range(1, x_range)]
                ax.set_xticks(np.arange(1, x_range))
                ax.set_xticklabels(chord_labels, rotation=90, ha='right')
                ax.set_xlabel('Chord')

        plt.tight_layout()
        plt.show()
    def analyze_progression_position_component_one_plot(self):
        x_transposed = self.vectorized_progressions.T
        num_rows = x_transposed.shape[0]
        x_range = 13  # chord range from 1 to 12

        fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(num_rows):
            sns.kdeplot(x_transposed[i, :], ax=ax, label=f'Position {i + 1}', bw_adjust=0.5)

        ax.set_xlim(1, 12)
        ax.set_title('Chord Progression Density by Position')
        ax.set_xlabel('Chord')
        ax.set_ylabel('Density')

        # Set x-ticks to chord labels
        chord_labels = [number_to_chord_label(n) for n in range(1, x_range)]
        ax.set_xticks(np.arange(1, x_range))
        ax.set_xticklabels(chord_labels, rotation=90, ha='right')

        ax.legend()
        plt.tight_layout()
        plt.show()

    def analyze_progression_component(self):
        x_transposed = self.vectorized_progressions.T
        num_rows = x_transposed.shape[0]
        x_range = 13  # chord range from 1 to 12

        fig = plt.figure(figsize=(8, 6))  # Adjusted figure size for better visibility
        gs = GridSpec(num_rows + 1, 1, height_ratios=[1] * num_rows + [2])

        # Plotting the overall heatmap
        overall_heatmap_data = np.zeros((num_rows, x_range - 1))
        for i in range(num_rows):
            for j in range(x_transposed.shape[1]):
                val = int(x_transposed[i, j])
                if 1 <= val < x_range:
                    overall_heatmap_data[i, val - 1] += 1  # Increment the count

        ax_heatmap = fig.add_subplot(gs[:-1])
        sns.heatmap(overall_heatmap_data, ax=ax_heatmap, cbar=False, cmap="binary", annot=False, linewidths=0.5,
                    linecolor='black')
        ax_heatmap.set_title('Overall Chord Sequence Distribution')
        ax_heatmap.set_ylabel('Chord Progression Index')

        # Set x-ticks to chord labels
        chord_labels = [number_to_chord_label(n) for n in range(1, x_range)]
        #ax_heatmap.set_xticks(np.arange(x_range - 1) + 0.5)
        #ax_heatmap.set_xticklabels(chord_labels, rotation=90, ha='right')

        # Plotting the overall KDE
        overall_data = x_transposed.flatten()
        ax_kde = fig.add_subplot(gs[-1])
        sns.kdeplot(overall_data, ax=ax_kde, color="black", bw_adjust=0.5, fill=True)
        ax_kde.set_xlim(1, 12)
        ax_kde.set_title('Density (Overall Chord Progression)')
        ax_kde.set_xlabel('Chord Label')
        ax_kde.set_ylabel('Density')

        # Set x-ticks to chord labels for KDE plot
        ax_kde.set_xticks(np.arange(1, x_range))
        ax_kde.set_xticklabels(chord_labels, rotation=90, ha='right')

        plt.tight_layout()
        plt.show()
