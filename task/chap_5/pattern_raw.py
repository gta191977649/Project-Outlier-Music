from sf_segmenter.segmenter import Segmenter
import librosa
import miditoolkit
import matplotlib.pyplot as plt
from feature.dataset import *
import feature.chord as chord
import numpy as np
from tslearn.clustering import silhouette_score
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d


#MODEL = TimeSeriesKMeans
MODEL = KMeans

def plotHarmonicProgression(harmonic_progression):
    values = chord.extractChordNumeralValues(harmonic_progression)
    plt.figure(figsize=(5, 2))
    plt.step(range(len(values)), values, where='mid', color='blue', linewidth=2)
    plt.title('Harmonic Pattern')
    plt.xlabel('Chord Progression Step')
    plt.ylabel('Chord Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def extractChordProgressionLabels(song : Song):
    chordProgressionLabels = []
    for chord in song.chord:
        time,beat,label = chord
        if not label == "N":
            chordProgressionLabels.append(label)
    return chordProgressionLabels

def normalize_to_max_length(X_train):
    # Find the maximum length of the sequences in X_train
    max_length = max(len(seq) for seq in X_train)

    # Create a new array to hold the normalized sequences
    normalized_X_train = np.zeros((len(X_train), max_length))

    # Fill the new array with the original sequences and pad with zeros
    for i, seq in enumerate(X_train):
        normalized_X_train[i, :len(seq)] = seq

    return normalized_X_train


def stretch_to_max_length(X_train):
    # Find the maximum length of the sequences in X_train
    max_length = max(len(seq) for seq in X_train)

    # Create a new array to hold the stretched sequences
    stretched_X_train = np.zeros((len(X_train), max_length))

    # Stretch each sequence to the max length using interpolation
    for i, seq in enumerate(X_train):
        original_indices = np.linspace(0, 1, len(seq))
        stretched_indices = np.linspace(0, 1, max_length)
        interpolator = interp1d(original_indices, seq, kind='linear')
        stretched_X_train[i] = interpolator(stretched_indices)

    return stretched_X_train

def eval_silhouette_score(X):
    scores = []
    Sum_of_squared_distances = []
    K = range(2, 15)
    for k in K:
        kmeans = MODEL(n_clusters=k,  random_state=0)
        km = kmeans.fit(X)
        # labels = km.predict(X_train)

        labels = km.labels_
        score = silhouette_score(X, labels, metric='euclidean')
        Sum_of_squared_distances.append(km.inertia_)
        print(score)
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

def plot_clusters(X_train, labels, centroids, k):
    fig, axes = plt.subplots(nrows=k, ncols=1,
                             figsize=(10, 2 * k))  # Adjust the figure size dynamically based on the number of clusters
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        if k == 1:
            ax = axes
        else:
            ax = axes[i]
        for j in cluster_indices:
            ax.plot(X_train[j], alpha=0.5, color='grey')  # Plot non-centroid series in grey
        ax.plot(centroids[i], color='red')  # Plot the centroid in red
        ax.set_title(f"Cluster {i + 1} ({len(cluster_indices)} matched)")
        ax.set_xlabel("Time")
        ax.set_ylabel("TPS")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # harmonic_progression = ['Fmaj', 'Cmaj', 'Gmaj', 'Amin']

    songs = loadSongCollection("/Users/nurupo/Desktop/dev/music4all/test_sample",filter="major")

    chord_singals = []

    for target_song in songs:
        chords = extractChordProgressionLabels(target_song)
        x = chord.extractChordNumeralValues(chords)
        chord_singals.append(x)
        #plotHarmonicProgression(chord_singal)

    # Cluster
    X_train = stretch_to_max_length(chord_singals)
    eval_silhouette_score(X_train)

    k = 7
    kmeans = MODEL(n_clusters=k, random_state=0)
    km = kmeans.fit(X_train)
    centroids = kmeans.cluster_centers_
    # labels = kmeans.predict(X_train)

    labels = km.labels_
    # output clustered songs sections
    # for i in range(k):
    #     print(f"Result for Cluster:{i + 1}")
    #     print(f"-------------------")
    #     for j in range(len(labels)):
    #         if labels[j] == i:
    #             print(Y_songs[j], Y_chord_progressions[j])
    #             print(Y_songs[j], X_train[j])
    #     print(f"-------------------")

    plot_clusters(X_train, labels, centroids, k=k)