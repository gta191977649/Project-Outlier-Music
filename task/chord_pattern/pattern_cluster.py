import os
import numpy as np
import feature.extract as extract
from model.song import Song
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA,LatentDirichletAllocation
import matplotlib.cm as cm
from tslearn.preprocessing import TimeSeriesScalerMinMax
if __name__ == '__main__':
    TARGET_MODE = "minor"
    TARGET_SECTION = "solo"
    PATH = "/Users/nurupo/Desktop/dev/music4all/europe/"
    # loop all folder
    song_collections = []
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith(".h5"):
                filepath = os.path.join(root, file)
                filename = os.path.splitext(file)[0]
                song = Song.from_h5(filepath)
                if song.mode == TARGET_MODE:
                    song_collections.append(song)

    # Prepare pattern dataset ...
    X_train = []
    for song in song_collections:
        for pat in song.chord_pattern:
            progression = pat["pattern"]
            thetas = []
            for chord in progression:
                # Do Chord Transpose Due to Patterns are on original key
                chord = extract.transposeChordLabel(chord,song.transpose_amount)
                angle = extract.getChordVectorsAngleFromChord(chord)
                thetas.append(angle)
            X_train.append(thetas)
    X_train = np.array(X_train)
    print(X_train.shape)

    # Determine n of clusters
    # Sum_of_squared_distances = []
    # K = range(2, 15)
    # for k in K:
    #     print(f"test k={k}")
    #     km = TimeSeriesKMeans(n_clusters=k,
    #                           n_init=2,
    #                           metric="dtw",
    #                           verbose=False,
    #                           max_iter_barycenter=10,
    #                           random_state=0)
    #
    #     km = km.fit(X_train)
    #     Sum_of_squared_distances.append(km.inertia_)
    # plt.plot(K, Sum_of_squared_distances, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Sum_of_squared_distances')
    # plt.title('Elbow Method For Optimal k')
    # plt.show()

    # train_pca = list(X_train.reshape(X_train.shape[0], X_train.shape[1]))
    # pca = PCA(n_components="mle")
    # train_pca = pca.fit_transform(train_pca)
    #
    # X = train_pca
    #
    # range_n_clusters = [3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16]
    # for n_clusters in range_n_clusters:
    #     # Create a subplot with 1 row and 2 columns
    #     fig, (ax1, ax2) = plt.subplots(1, 2)
    #     fig.set_size_inches(18, 7)
    #
    #     # The 1st subplot is the silhouette plot
    #     # The silhouette coefficient can range from -1, 1 but in this example all
    #     # lie within [-0.1, 1]
    #     ax1.set_xlim([-0.1, 1])
    #     # The (n_clusters+1)*10 is for inserting blank space between silhouette
    #     # plots of individual clusters, to demarcate them clearly.
    #     ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    #
    #     # Initialize the clusterer with n_clusters value and a random generator
    #     # seed of 10 for reproducibility.
    #     clusterer = KMeans(n_clusters=n_clusters, random_state=0)
    #     cluster_labels = clusterer.fit_predict(X)
    #
    #     # The silhouette_score gives the average value for all the samples.
    #     # This gives a perspective into the density and separation of the formed
    #     # clusters
    #     silhouette_avg = silhouette_score(X, cluster_labels)
    #     print(
    #         "For n_clusters =",
    #         n_clusters,
    #         "The average silhouette_score is : {:.2f}"
    #         .format(silhouette_avg),
    #     )
    #
    #     # Compute the silhouette scores for each sample
    #     sample_silhouette_values = silhouette_samples(X, cluster_labels)
    #
    #     y_lower = 10
    #     for i in range(n_clusters):
    #         # Aggregate the silhouette scores for samples belonging to
    #         # cluster i, and sort them
    #         ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    #         ith_cluster_silhouette_values.sort()
    #
    #         size_cluster_i = ith_cluster_silhouette_values.shape[0]
    #         y_upper = y_lower + size_cluster_i
    #
    #         color = cm.nipy_spectral(float(i) / n_clusters)
    #         ax1.fill_betweenx(
    #             np.arange(y_lower, y_upper),
    #             0,
    #             ith_cluster_silhouette_values,
    #             facecolor=color,
    #             edgecolor=color,
    #             alpha=0.7,
    #         )
    #
    #         # Label the silhouette plots with their cluster numbers at the middle
    #         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    #
    #         # Compute the new y_lower for next plot
    #         y_lower = y_upper + 10  # 10 for the 0 samples
    #
    #     ax1.set_title("The silhouette plot for the various clusters.")
    #     ax1.set_xlabel("The silhouette coefficient values")
    #     ax1.set_ylabel("Cluster label")
    #
    #     # The vertical line for average silhouette score of all the values
    #     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    #
    #     ax1.set_yticks([])  # Clear the yaxis labels / ticks
    #     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    #
    #     # 2nd Plot showing the actual clusters formed
    #     colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    #     ax2.scatter(
    #         X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    #     )
    #
    #     # Labeling the clusters
    #     centers = clusterer.cluster_centers_
    #     # Draw white circles at cluster centers
    #     ax2.scatter(
    #         centers[:, 0],
    #         centers[:, 1],
    #         marker="o",
    #         c="white",
    #         alpha=1,
    #         s=200,
    #         edgecolor="k",
    #     )
    #
    #     for i, c in enumerate(centers):
    #         ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
    #
    #     ax2.set_title("The visualization of the clustered data.")
    #     ax2.set_xlabel("Feature space for the 1st feature")
    #     ax2.set_ylabel("Feature space for the 2nd feature")
    #
    #     plt.suptitle(
    #         "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
    #         % n_clusters,
    #         fontsize=14,
    #         fontweight="bold",
    #     )
    #
    # plt.show()




    # Do Time Series Clustering
    n_clusters = 14
    dba_km = TimeSeriesKMeans(n_jobs=8,n_clusters=n_clusters,
                              n_init=1,
                              metric="dtw",
                              verbose=False,
                              max_iter_barycenter=5,
                              random_state=1)
    y_pred_dba_km = dba_km.fit_predict(X_train)

    sz = X_train.shape[1]

    plt.figure(figsize=(10, n_clusters * 3))
    for yi in range(n_clusters):
        plt.subplot(n_clusters, 1, yi + 1)
        for xx in X_train[y_pred_dba_km == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, sz)
        #plt.ylim(-4, 4)
        plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                 transform=plt.gca().transAxes)
        if yi == 1:
            plt.title("DBA $k$-means")

    plt.tight_layout()
    plt.show()