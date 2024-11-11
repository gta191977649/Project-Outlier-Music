# The oldest but easiler implmentation
from model.song import Song
from feature.chord import *
from feature.dataset import *
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tslearn.clustering import silhouette_score
from sklearn.cluster import KMeans
from collections import defaultdict

from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import seaborn as sns
from model.isolation_tree import *
from sklearn.decomposition import PCA
import umap

MODEL = KMeans




def eval_silhouette_score(X, max=None):
    scores = []
    Sum_of_squared_distances = []
    max = max or int(len(X) / 2)
    K = range(2, max)
    for k in K:
        kmeans = MODEL(n_init=10, n_clusters=k, random_state=0)
        km = kmeans.fit(X)
        # labels = km.predict(X_train)

        labels = km.labels_
        score = silhouette_score(X, labels, metric='euclidean')
        Sum_of_squared_distances.append(km.inertia_)
        print(score, k)
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


def eval_best_k(X, min_clusters=2, max_clusters=None, convergence_threshold=0.001):
    """
    Evaluate and determine the optimal number of clusters for KMeans using inertia.
    Stop when the improvement in inertia falls below the convergence threshold.

    Parameters:
    - X: The data to be clustered.
    - min_clusters: Minimum number of clusters to evaluate (default: 2).
    - max_clusters: Maximum number of clusters to evaluate (default: half of the dataset size).
    - convergence_threshold: Minimum improvement in inertia to continue evaluation (default: 0.01).

    Returns:
    - best_k: The optimal number of clusters based on inertia convergence.
    """
    inertia_values = []

    # Determine maximum number of clusters if not provided
    if max_clusters is None:
        max_clusters = int(len(X) / 2)

    best_k = min_clusters
    prev_inertia = None

    # Evaluate inertia for each k
    for k in range(min_clusters, max_clusters):
        kmeans = KMeans(n_init=10, n_clusters=k, random_state=0)
        km = kmeans.fit(X)
        inertia = km.inertia_
        inertia_values.append(inertia)

        # Check for convergence based on improvement threshold
        if prev_inertia is not None and (prev_inertia - inertia) < convergence_threshold:
            break

        prev_inertia = inertia
        best_k = k

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(min_clusters, min_clusters + len(inertia_values)), inertia_values, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.title('Inertia for Different k Values')
    plt.axvline(x=best_k, color='green', linestyle='--', label=f'Optimal k={best_k}')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    print(f"Optimal number of clusters: {best_k}")

    return best_k


def plot_data(data):
    plt.figure(figsize=(8, 6))
    for i in range(data.shape[0]):
        plt.scatter(range(1, 5), data[i, :],
                    label=f"Data {i + 1}" if i < 5 else "")  # Plot only first 5 labels for clarity

    plt.show()
    plt.figure(figsize=(10, 6))
    for i in range(data.shape[1]):  # Iterate over each column
        sns.kdeplot(data[:, i], label=f"Index {i + 1}")

    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Density Plot of Data by Index")
    plt.legend(title="Index")
    plt.grid(True)
    plt.show()


def plot_clusters(X_train, labels, centroids, k, style="default"):
    # Calculate the size of each cluster
    cluster_sizes = [(i, np.sum(labels == i)) for i in range(k)]

    # Sort clusters by size in descending order
    sorted_clusters = sorted(cluster_sizes, key=lambda x: x[1], reverse=True)

    fig, axes = plt.subplots(nrows=k, ncols=1, figsize=(10, 2 * k))

    for plot_idx, (i, size) in enumerate(sorted_clusters):
        cluster_indices = np.where(labels == i)[0]
        ax = axes[plot_idx] if k > 1 else axes  # Handle case when k=1
        for j in cluster_indices:
            ax.plot(X_train[j], alpha=0.5, drawstyle=style, color='grey')
        ax.plot(centroids[i], drawstyle=style, color='red')
        ax.set_title(f"Cluster {i + 1} ({size} matched)")
        ax.set_xlabel("Time")
        ax.set_ylabel("TPS")

    plt.tight_layout()
    plt.show()


def sort_pattern_by_cluster_frequency(chord_signals, labels):
    """
    Save the patterns sorted by frequency in each cluster in the same format as X_train.

    Parameters:
    - chord_signals: A list of chord numeral patterns.
    - labels: A list of cluster labels corresponding to each chord pattern.

    Returns:
    - sorted_patterns: A numpy array with patterns sorted by frequency of patterns in each cluster.
    """
    # Create a dictionary to hold chord patterns by cluster
    cluster_chords = defaultdict(list)

    # Organize chord patterns by cluster
    for idx, label in enumerate(labels):
        cluster_chords[label].append(chord_signals[idx])

    # Sort clusters by size (number of patterns) in descending order
    sorted_clusters = sorted(cluster_chords.items(), key=lambda x: len(x[1]), reverse=True)

    # Flatten the sorted list of patterns
    sorted_patterns = [pattern for _, patterns in sorted_clusters for pattern in patterns]

    # Convert the list of patterns to a numpy array
    sorted_patterns_array = np.array(sorted_patterns)

    return sorted_patterns_array


def plot_pattern_frequency(chord_signals, labels):
    """
    Plot a bar chart showing the frequency of patterns in each cluster.

    Parameters:
    - chord_signals: A list of chord numeral patterns.
    - labels: A list of cluster labels corresponding to each chord pattern.
    """
    # Create a dictionary to hold chord patterns by cluster
    cluster_chords = defaultdict(list)

    # Organize chord patterns by cluster
    for idx, label in enumerate(labels):
        cluster_chords[label].append(chord_signals[idx])

    # Sort clusters by size (number of patterns) in descending order
    sorted_clusters = sorted(cluster_chords.items(), key=lambda x: len(x[1]), reverse=True)

    # Prepare data for plotting
    cluster_names = []
    pattern_counts = []

    for cluster_id, patterns in sorted_clusters:
        num_patterns = len(patterns)
        pattern_counts.append(num_patterns)

        # Use the first pattern as the cluster name
        first_pattern = patterns[0]
        chord_labels = translateNumeralValuesToChords(first_pattern)
        cluster_name = " ".join(chord_labels)
        cluster_names.append(cluster_name)

    # Plot the bar chart
    plt.figure(figsize=(25, 5))
    plt.bar(range(len(pattern_counts)), pattern_counts, tick_label=cluster_names, align='center', color='blue',
            width=0.8)
    plt.xticks(rotation=90)
    plt.xlabel('Cluster (First Chord Pattern)')
    plt.ylabel('Number of Patterns')
    plt.title('Number of Patterns in Each Cluster')
    plt.tight_layout()
    plt.show()


def print_chord_patterns_by_cluster(chord_signals, labels, num_clusters, mode="major"):
    # Create a dictionary to hold chord patterns by cluster
    cluster_chords = defaultdict(list)

    # Organize chord patterns by cluster
    for idx, label in enumerate(labels):
        cluster_chords[label].append(chord_signals[idx])

    # Sort clusters by size (number of patterns) in descending order
    sorted_clusters = sorted(cluster_chords.items(), key=lambda x: len(x[1]), reverse=True)

    # Print the chord patterns and labels for each cluster
    for cluster_id, patterns in sorted_clusters:
        print(f"\nCluster {cluster_id + 1} (Number of patterns: {len(patterns)}):")
        for signal in patterns:
            chord_labels = translateNumeralValuesToChords(signal, mode=mode)
            print(f"Chord Pattern (Numerals): {signal}")
            print(f"Chord Labels: {chord_labels}")


def run_detection_with_all(mode):
    songs = loadSongCollection(r"/Users/nurupo/Desktop/dev/music4all/h5_pop_all", mode=mode)
    # songs = loadSongCollection(r"/Users/nurupo/Desktop/dev/audio/aimyon", mode="major")
    chord_signals = []
    chord_labels = []
    cadece_consider = {}
    cadece_consider["major"] = [
        ["G:maj", "C:maj"],  # Perfect Cadence
        ["F:maj", "C:maj"],  # Plagal Cadence
        ["C:maj", "G:maj"],  # Half Cadence
        ["D:maj", "G:maj"],  # Half Cadence
        ["F:maj", "G:maj"],  # Half Cadence
        ["G:maj", "A:min"],  # Deceptive Cadence
    ]

    cadece_consider["minor"] = [
        ["E:min", "A:min"],  # Perfect Cadence
        ["D:min", "A:min"],  # Plagal Cadence
        ["A:min", "E:min"],  # Half Cadence
        ["B:dim", "E:min"],  # Half Cadence
        ["D:min", "E:min"],  # Half Cadence
        ["E:min", "F#:maj"],  # Deceptive Cadence
    ]
    for target_song in songs:
        chords = target_song.extractChordProgressionLabels(transposed=True)
        x = extractChordNumeralValues(chords)
        x = filterRepeatSignal(x)
        chords = filterRepeatSignal(chords)

        for cadece in cadece_consider[mode]:
            cadence_signal = extractChordNumeralValues(cadece)
            matches = find_cadence_patterns(x, cadence_signal, min_preceding_chords=2)
            for start, end in matches:
                chord_signals.append(x[start:end])
                label = convert_roman_label(chords[start:end], mode=mode)
                chord_labels.append(label)
                # chord_labels.append(chords[start:end])
            # plotHarmonicProgression(chord_singal)

    # X_train = stretch_to_max_length(chord_singals)
    X_train = np.array(chord_labels)  # The Tokenization processing is done in ChordProgressionAnalyzer
    # X_train = X_train.T[0]
    isolation_tree = ChordProgressionIsolationForest(X_train, tokenMethod="onehot", dimReduceMethod=None)  # numeral

    outliers = isolation_tree.detect_outliers()
    print("Outlier indices:", np.where(outliers == -1)[0])

    #isolation_tree.visualize_progression_tree_hierarchical(outliers_only=True)
    # isolation_tree.plot_result(annOutlier=True)
    # isolation_tree.visualize_anomaly_scores()

    isolation_tree.print_outliers()
    isolation_tree.save_outliers("outliers_all.csv")

    #isolation_tree.visualize_progression_dendrogram(outliers_only=True)
    # isolation_tree.visualize_in_web(horizontal_spacing=18,node_size_scale=5,font_size=15)
    print(f"Outliers:{len(np.where(outliers == -1)[0])} of {len(X_train)} Data")
def run_detection_positional(mode,position):
    songs = loadSongCollection(r"/Users/nurupo/Desktop/dev/music4all/h5_pop_all", mode=mode)
    #songs = loadSongCollection(r"/Users/nurupo/Desktop/dev/audio/aimyon", mode="major")

    chord_signals = []
    chord_labels = []
    cadece_consider = {}
    cadece_consider["major"] = [
        ["G:maj", "C:maj"],  # Perfect Cadence
        ["F:maj", "C:maj"],  # Plagal Cadence
        ["C:maj", "G:maj"],  # Half Cadence
        ["D:maj", "G:maj"],  # Half Cadence
        ["F:maj", "G:maj"],  # Half Cadence
        ["G:maj", "A:min"],  # Deceptive Cadence
    ]

    cadece_consider["minor"] = [
        ["E:min", "A:min"],  # Perfect Cadence
        ["D:min", "A:min"],  # Plagal Cadence
        ["A:min", "E:min"],  # Half Cadence
        ["B:dim", "E:min"],  # Half Cadence
        ["D:min", "E:min"],  # Half Cadence
        ["E:min", "F#:maj"],  # Deceptive Cadence
    ]
    for target_song in songs:
        chords = target_song.extractChordProgressionLabels(transposed=True)
        x = extractChordNumeralValues(chords)
        x = filterRepeatSignal(x)
        chords = filterRepeatSignal(chords)

        for cadece in cadece_consider[mode]:
            cadence_signal = extractChordNumeralValues(cadece)
            matches = find_cadence_patterns(x, cadence_signal, min_preceding_chords=2)
            for start, end in matches:
                chord_signals.append(x[start:end])
                label = convert_roman_label(chords[start:end], mode=mode)
                chord_labels.append(label)
                #chord_labels.append(chords[start:end])
            # plotHarmonicProgression(chord_singal)

    # X_train = stretch_to_max_length(chord_singals)
    X_train = np.array(chord_labels)  # The Tokenization processing is done in ChordProgressionAnalyzer

    X_train = X_train.T[position]

    #print(X_train)
    isolation_tree = ChordProgressionIsolationForest(X_train, tokenMethod="onehot", dimReduceMethod=None)  # numeral

    outliers = isolation_tree.detect_outliers()
    print("Outlier indices:", np.where(outliers == -1)[0])

    #isolation_tree.plot_result(annOutlier=True)
    #isolation_tree.visualize_anomaly_scores()


    #isolation_tree.visualize_progression_tree_hierarchical()
    #isolation_tree.visualize_in_web(horizontal_spacing=10,node_size_scale=10,font_size=15)
    #isolation_tree.print_outliers()
    isolation_tree.save_outliers(f"outliers_pos_{position}.csv")

if __name__ == '__main__':
    run_detection_with_all("major")
    #run_detection_positional("major",0)