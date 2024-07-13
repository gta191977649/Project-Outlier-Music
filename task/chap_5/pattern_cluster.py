from model.song import Song
import os
import feature.pattern as patternFeature
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from tslearn.clustering import silhouette_score

def loadSongCollection(PATH):
    song_collections = []
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith(".h5"):
                filepath = os.path.join(root, file)
                filename = os.path.splitext(file)[0]
                song = Song.from_h5(filepath)
                song_collections.append(song)
    return song_collections

def eval_silhouette_score(X):
    scores = []
    Sum_of_squared_distances = []
    K = range(2, 15)
    for k in K:
        kmeans = KMeans(n_clusters=k,  random_state=0)
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
    TARGET_SECTION = "chorus"

    corpus = loadSongCollection("/Users/nurupo/Desktop/dev/audio/akb48/")

    song_pattern = []

    for song in corpus:
        tonicKey = f"{song.key}:{song.mode[:3]}"

        pattern_singal = patternFeature.extractTontalPitchDistancePattern(song.chord_pattern[0]["pattern"], key=tonicKey,mode="profile")
        song_pattern.append(pattern_singal)
        # for feature in song.chord_pattern:
        #     if len(feature["pattern"]) > 0:
        #         pattern_singal = patternFeature.extractTontalPitchDistancePattern(feature["pattern"],key=tonicKey,mode="profile")
        #         song_pattern.append(pattern_singal)

    print(song_pattern)
    song_pattern = np.array(song_pattern)
    eval_silhouette_score(song_pattern)

    k = 7
    model = KMeans(n_clusters=k,  random_state=0)

    km = model.fit(song_pattern)
    centroids = model.cluster_centers_
    labels = km.labels_
    # output clustered songs sections
    for i in range(k):
        print(f"Result for Cluster:{i + 1}, target section: {TARGET_SECTION}")
        print(f"-------------------")
        for j in range(len(labels)):
            if labels[j] == i:
                print(j, song_pattern[j])
        print(f"-------------------")

    plot_clusters(song_pattern, labels, centroids, k=k)