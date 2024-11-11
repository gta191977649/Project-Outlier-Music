from feature.chord import *
from feature.dataset import *
import feature.pattern as pattern
import numpy as np
import seaborn as sns
import librosa
import matplotlib.pyplot as plt
from tslearn.clustering import silhouette_score
from sklearn.cluster import KMeans
from collections import defaultdict
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

def plot_chord_sequence_heatmap(X_train):
    """
       Plot a heatmap to visualize the chord sequence patterns in X_train.

       Parameters:
       - X_train: A (2405, 4) array representing chord sequences.
       """
    # Define the chord pitch values based on the provided mapping
    chord_pitches = [1,2,3, 2.5, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7]


    # Create a custom color palette for the chord pitch values
    colors = sns.color_palette("viridis", len(chord_pitches) - 1)
    cmap = ListedColormap(colors)

    # Create the heatmap using seaborn with the chord pitch as color
    sns.heatmap(
        X_train,
        cmap=cmap,  # Use the custom colormap
    )

    # Set axis labels and title
    plt.xlabel('Chord Sequence Index')
    plt.ylabel('Pattern Index')
    plt.title('Heatmap of Chord Sequences by Chord Pitch Value')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Display the plot
    plt.show()

def plot_chord_pattern_3d(X_train):
    """
       Plot a 3D plot with projected filled contours and connected curves
       to visualize the chord sequence patterns in X_train.

       Parameters:
       - X_train: A (2405, 4) array representing chord sequences.
       """
    # Define the number of patterns and sequence indices
    num_patterns, num_indices = X_train.shape

    # Define the chord pitch values and corresponding labels
    chord_pitches = [1,2,3,4,5,6,7,8,9,10,11,12]
    chord_labels = ["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"]

    # Create a custom color palette for chord pitch values
    colors = sns.color_palette("Paired", len(chord_pitches))  # 13 distinct colors
    cmap = ListedColormap(colors)

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare data for the 3D plot
    for pattern_index in range(num_patterns):
        x_data = np.full(num_indices, pattern_index)  # Pattern index repeated
        y_data = np.arange(1, num_indices + 1)  # Chord sequence index
        z_data = X_train[pattern_index]  # Chord pitch values for the current pattern

        # Map each pitch to an index for the color palette
        color_indices = np.array([chord_pitches.index(pitch) for pitch in z_data])

        # Plot lines to connect the points in each pattern
        ax.plot(x_data, y_data, z_data, color='blue', alpha=0.5)

        # Plot the points with color according to pitch value
        ax.scatter(x_data, y_data, z_data, c=color_indices, cmap=cmap, marker='o')

    # Create a grid for contour plots
    X, Y = np.meshgrid(np.arange(num_patterns), np.arange(1, num_indices + 1))
    Z = np.zeros_like(X, dtype=float)

    # Compute the mean pitch value for contour plotting
    for i in range(num_patterns):
        for j in range(num_indices):
            Z[j, i] = np.mean(X_train[i, :])

    # Plot projected filled contours
    ax.contourf(X, Y, Z, zdir='z', offset=0, cmap='viridis', alpha=0.3)  # Contour on the base

    # Set labels for the axes
    ax.set_xlabel('Pattern Index')
    ax.set_ylabel('Chord Sequence Index')
    ax.set_zlabel('Chord Pitch Value')
    ax.set_title('Harmonic Pattern')

    # Add a color bar
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(chord_pitches)  # For chord pitch values with step size 0.5
    cbar = fig.colorbar(mappable, ax=ax, pad=0.1, ticks=chord_pitches)
    cbar.set_label('Chord Pitch Value')
    cbar.set_ticks(chord_pitches)
    cbar.set_ticklabels(chord_labels)

    # Set the limits
    ax.set_xlim(0, num_patterns)
    ax.set_ylim(1, num_indices)
    ax.set_zlim(0, 12)

    # Display the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # songs = loadSongCollection(r"/Users/nurupo/Desktop/dev/audio/aimyon", mode="major")
    #
    # chord_signals = []
    # chord_labels = []
    # cadence_consider = [
    #     ["G:maj", "C:maj"],  # Perfect Cadence
    #     ["F:maj", "C:maj"],  # Plagal Cadence
    #     ["C:maj", "G:maj"],  # Half Cadence
    #     ["D:maj", "G:maj"],  # Half Cadence
    #     ["F:maj", "G:maj"],  # Half Cadence
    #     ["G:maj", "A:min"],  # Deceptive Cadence
    # ]
    # for target_song in songs:
    #     chords = target_song.extractChordProgressionLabels(transposed=True)
    #     x = extractChordNumeralValues(chords)
    #     x = filterRepeatSignal(x)
    #     chords = filterRepeatSignal(chords)
    #
    #     # do cadence match
    #     for cadence in cadence_consider:
    #         cadence_signal = extractChordNumeralValues(cadence)
    #         matches = pattern.find_cadence_patterns(x, cadence_signal, min_preceding_chords=2)
    #         for start, end in matches:
    #             chord_signals.append(x[start:end])
    #             chord_labels.append(chords[start:end])

    #X_train = np.array(chord_signals)
    X_train = np.load(f"./pattern.npy")

    # Plot the heatmap
    plot_chord_sequence_heatmap(X_train)
    #plot_chord_pattern_3d(X_train)

    print(X_train)
    print(X_train.shape)
