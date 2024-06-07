from sklearn.cluster import KMeans
import numpy as np
from model.song import Song
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import feature.extract as extract

from scipy.cluster.hierarchy import linkage, dendrogram
from dtaidistance import dtw
from scipy.spatial.distance import pdist, squareform

from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA, LatentDirichletAllocation
import matplotlib.cm as cm
from tslearn.preprocessing import TimeSeriesScalerMinMax
import feature.pattern as patternFeature
import os
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering

MODEL = TimeSeriesKMeans


# MODEL = KMeans

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


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


def eval_silhouette_score(X):
    scores = []
    Sum_of_squared_distances = []
    K = range(2, 10)
    for k in K:
        kmeans = MODEL(n_clusters=k, metric="dtw", random_state=0)
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


def find_section_label(time, s):
    for section in s:
        if section['start'] >= time and time < section['end']:
            return section['label']
    return False


def isTimmingInSection(time, sectionName, s, matchFirst=True):
    section_timming = None
    for section in s:
        if section["label"] == sectionName:
            section_timming = [float(section['start']), float(section['end'])]
            if matchFirst: break  # break if we only want to first match
    if section_timming:
        if float(time) >= section_timming[0] and float(time) <= section_timming[1]:
            return True
    return False


def normalize_to_max_length(X_train):
    # Find the maximum length of the sequences in X_train
    max_length = max(len(seq) for seq in X_train)

    # Create a new array to hold the normalized sequences
    normalized_X_train = np.zeros((len(X_train), max_length))

    # Fill the new array with the original sequences and pad with zeros
    for i, seq in enumerate(X_train):
        normalized_X_train[i, :len(seq)] = seq

    return normalized_X_train


if __name__ == '__main__':
    #TARGET_MODE = "major"
    TARGET_SECTION = "chorus"
    PATH = "/Users/nurupo/Desktop/dev/music4all/custom"
    print(TARGET_SECTION)
    # loop all folder

    song_collections = []
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith(".h5"):
                filepath = os.path.join(root, file)
                filename = os.path.splitext(file)[0]
                song = Song.from_h5(filepath)
                song_collections.append(song)

    # Prepare pattern dataset
    X_train = []
    Y_songs = []
    Y_chord_progressions = []
    Y_timming = []
    for song in song_collections:
        # for pat in song.chord_pattern:
        #     progression = pat["pattern"]
        #     signal = patternFeature.extractTontalPitchDistancePattern(progression)
        #     X_train.append(signal)

        sections = song.section
        chord_progression = []
        times = []
        for i in range(len(song.chord)):
            time, beat, chord = song.chord[i]
            chord = chord.replace(":", "")
            # Todo: ensuring beat alinment is correct
            if isTimmingInSection(float(time), TARGET_SECTION, sections):
                # if find_section_label(float(time), sections) == TARGET_SECTION: #if matched section is found
                chord_progression.append(chord)
                times.append(time)

        if len(chord_progression) < 3:
            print(chord_progression)
        #key = f"{song.key}:{song.mode[:3]}"
        # NOPE! We should take the first key from chord progression as home key instead!
        key = chord_progression[0]
        print(key)
        signal = patternFeature.extractTontalPitchDistancePattern(chord_progression, key, mode="profile")
        # signal = patternFeature.extractChromaticPattern(chord_progression)
        # if len(signal) >0:
        X_train.append(signal)
        Y_songs.append(song.title)
        Y_timming.append(times)
        Y_chord_progressions.append(chord_progression)

    print(X_train)
    X_train = normalize_to_max_length(X_train)
    X_train = np.array(X_train)
    #eval_silhouette_score(X_train)

    k =3
    kmeans = MODEL(n_clusters=k, metric="dtw", random_state=0)
    km = kmeans.fit(X_train)
    centroids = kmeans.cluster_centers_
    # labels = kmeans.predict(X_train)

    labels = km.labels_
    # output clustered songs sections
    for i in range(k):
        print(f"Result for Cluster:{i + 1}, target section: {TARGET_SECTION}")
        print(f"-------------------")
        for j in range(len(labels)):
            if labels[j] == i:
                print(Y_songs[j], Y_chord_progressions[j])
        print(f"-------------------")

    plot_clusters(X_train, labels, centroids, k=k)
