# The oldest but easiler implmentation
from model.song import Song
from feature.chord import *
from feature.dataset import *
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tslearn.clustering import silhouette_score
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d

MODEL = KMeans
def find_cadence_patterns(main_signal, cadence_pattern, min_preceding_chords=2, allow_repetitions=True):
    """
    Find multiple occurrences of a cadence pattern in the main signal using exact matching,
    with an option to allow or disallow repetitive chords.

    Parameters:
    - main_signal: The main chord progression signal (list of numbers)
    - cadence_pattern: The cadence pattern to search for (list of numbers)
    - min_preceding_chords: Minimum number of chords required before the cadence pattern (default: 2)
    - allow_repetitions: Whether to allow repetitive chords in the progression (default: True)

    Returns:
    - A list of tuples, each containing (start_index, end_index) of found patterns
    """
    pattern_length = len(cadence_pattern)
    matches = []

    for i in range(min_preceding_chords, len(main_signal) - pattern_length + 1):
        # Check if the cadence pattern matches exactly
        if main_signal[i:i + pattern_length] == cadence_pattern:
            # Check if there are enough preceding chords
            if i >= min_preceding_chords:
                # If repetitions are not allowed, check for unique chords
                if not allow_repetitions:
                    progression = main_signal[i - min_preceding_chords:i + pattern_length]
                    if len(set(progression)) == len(progression):
                        matches.append((i - min_preceding_chords, i + pattern_length))
                else:
                    matches.append((i - min_preceding_chords, i + pattern_length))

    return matches

def eval_silhouette_score(X):
    scores = []
    Sum_of_squared_distances = []
    K = range(2, int(len(X) / 2))
    for k in K:
        kmeans = MODEL(n_clusters=k,  random_state=0)
        km = kmeans.fit(X)
        # labels = km.predict(X_train)

        labels = km.labels_
        score = silhouette_score(X, labels, metric='euclidean')
        Sum_of_squared_distances.append(km.inertia_)
        print(score,k)
        scores.append(score)

    fig, ax1 = plt.subplots()

    color = 'blue'
    ax1.set_xlabel('k')
    ax1.set_ylabel('Silhouette Score', color=color)
    ax1.plot(K, scores, 'bx-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'red'
    ax2.set_ylabel('Sum of Squared Distances', color=color)
    ax2.plot(K, Sum_of_squared_distances, 'rx-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Silhouette Score and Sum of Squared Distances for k')
    fig.tight_layout()
    plt.show()
def plot_clusters(X_train, labels, centroids, k,style="default"):
    fig, axes = plt.subplots(nrows=k, ncols=1,
                             figsize=(10, 2 * k))  # Adjust the figure size dynamically based on the number of clusters
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        if k == 1:
            ax = axes
        else:
            ax = axes[i]
        for j in cluster_indices:
            ax.plot(X_train[j], alpha=0.5, drawstyle=style, color='grey')  # Plot non-centroid series in grey
        ax.plot(centroids[i],drawstyle=style, color='red')  # Plot the centroid in red
        ax.set_title(f"Cluster {i + 1} ({len(cluster_indices)} matched)")
        ax.set_xlabel("Time")
        ax.set_ylabel("TPS")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    songs = loadSongCollection(r"F:\music4all\sample_test", mode="major")

    chord_signals = []

    cadece_consider = [
        ["G:maj", "C:maj"], # Perfect Cadence
        ["F:maj", "C:maj"], # Plagal Cadence
        #["C:maj", "G:maj"], # Half Cadence
        ["G:maj", "A:min"], # Deceptive Cadence
    ]
    for target_song in songs:
        chords = target_song.extractChordProgressionLabels(transposed=True)
        x = extractChordNumeralValues(chords)
        x = filterRepeatSignal(x)

        # do candence match
        for cadece in cadece_consider:
            cadence_signal = extractChordNumeralValues(cadece)
            matches = find_cadence_patterns(x, cadence_signal, min_preceding_chords=2)
            for start, end in matches:
                chord_signals.append(x[start:end])
            # plotHarmonicProgression(chord_singal)


    #X_train = stretch_to_max_length(chord_singals)
    X_train = np.array(chord_signals)

    print(f"Total: {X_train.shape[0]}")
    eval_silhouette_score(X_train)

    k = 60
    kmeans = MODEL(n_clusters=k, random_state=0)
    km = kmeans.fit(X_train)
    centroids = kmeans.cluster_centers_
    # labels = kmeans.predict(X_train)

    labels = km.labels_
    print(f"N of centroids: {len(centroids)}")
    plot_clusters(X_train, labels, centroids, k=k,style="default")


def test():
    file = r"F:\music4all\pop_h5\0XTBHHzLg9mngdvU.h5"
    song = Song.from_h5(file)

    labels = song.extractChordProgressionLabels(transposed=True)
    signal = extractChordNumeralValues(labels)
    signal = filterRepeatSignal(signal)
    labels = filterRepeatSignal(labels)
    cadence_signal = extractChordNumeralValues([
        "G:maj", "A:min",  # perfect cadence V → I
    ])

    matches = find_cadence_patterns(signal, cadence_signal,min_preceding_chords=2)

    print(f"Chord Signal: {signal}")
    print(f"Chord Label: {labels}")
    print(f"Cadence Signal: {cadence_signal}")
    if matches:
        print(f"Found {len(matches)} cadence pattern(s):")
        for start, end in matches:
            print(f"  Position {start} to {end}: {labels[start:end]}")
    else:
        print("No cadence patterns found.")