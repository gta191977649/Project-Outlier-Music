import numpy as np
import matplotlib.pyplot as plt
import feature.msd_dataset as msd
import os
import feature.msd_getter as h5
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, multivariate_normal

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import mahalanobis
def group_files_by_release(directory):
    releases = {}
    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            full_path = os.path.join(directory, filename)
            song = h5.open_h5_file_read(full_path)
            release = h5.get_release(song).decode("utf-8")
            if release not in releases:
                releases[release] = []
            releases[release].append(full_path)
    return releases
def loadArtistCollection(directory):
    collection = []
    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            full_path = os.path.join(directory, filename)
            #song = h5.open_h5_file_read(full_path)
            collection.append(full_path)
    return collection
# Load MFCC features for the artist
# Load the average MFCC feature for each song of the artist
def group_files_by_release(directory):
    releases = {}
    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            full_path = os.path.join(directory, filename)
            song = h5.open_h5_file_read(full_path)
            release = h5.get_release(song).decode("utf-8")
            if release not in releases:
                releases[release] = []
            releases[release].append(full_path)
    return releases
def loadArtistCollection(directory):
    collection = []
    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            full_path = os.path.join(directory, filename)
            #song = h5.open_h5_file_read(full_path)
            collection.append(full_path)
    return collection
# Load MFCC features for the artist
# Load the average MFCC feature for each song of the artist
def load_mfcc_features(directory):
    collection = loadArtistCollection(directory)
    #collection = group_files_by_release(directory)
    mfcc_features = []
    for path in collection["Extraterrestrial Live"]:
        song = msd.openSong(path)
        tempo = msd.getFeature(song, feature="tempo")  # Shape (12,), representing one vector per song
        mfcc = msd.getFeature(song, feature="mfcc")  # Shape (12,), representing one vector per song
        mfcc_features.append(mfcc)
    return np.array(mfcc_features)
# Load tempo and loudness features for songs in a specific album
def load_tempo_loudness_for_album(directory, album_name="Extraterrestrial Live"):
    #collection = group_files_by_release(directory)
    collection = loadArtistCollection(directory)

    features = []
    for path in collection:
        song = msd.openSong(path)
        tempo = msd.getFeature(song, feature="tempo")
        loudness = msd.getFeature(song, feature="loudness")
        print(loudness)
        features.append([tempo, loudness])

    return np.array(features)
def loadMSDCsvData(path):
    songs = []
    csv = pd.read_csv(path)
    for idx,item in csv.iterrows():
        songs.append(item.to_dict())
    return songs
def get_artist_scatter_feature(artist, x="tempo", y="loudness", normalize=False):
    songs = loadMSDCsvData("/Users/nurupo/Desktop/dev/msd/song_v2.csv")
    df = pd.DataFrame(songs)
    df_artist = df[df['artist'] == artist]
    X = df_artist[x]
    Y = df_artist[y]

    if normalize:
        # Reshape X and Y to 2D arrays for the scaler
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X.values.reshape(-1, 1)).flatten()
        Y = scaler.fit_transform(Y.values.reshape(-1, 1)).flatten()

    features = list(zip(X, Y))

    return np.array(features)


def is_inside_gaussian(sample, mean, cov, threshold=1e-3):
    """Check if a sample is 'inside' the Gaussian component based on a PDF threshold."""
    pdf_value = multivariate_normal(mean=mean, cov=cov).pdf(sample)
    return pdf_value > threshold


def plot_contours(data, means, covs, title):
    """visualize the gaussian components over the data"""
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], 'ko')

    delta = 0.025
    k = means.shape[0]
    x = np.arange(-0.5, 1.5, delta)
    y = np.arange(-0.5, 1.5, delta)
    x_grid, y_grid = np.meshgrid(x, y)
    coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T

    col = ['green', 'red', 'indigo']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        z_grid = multivariate_normal(mean, cov).pdf(coordinates).reshape(x_grid.shape)
        plt.contour(x_grid, y_grid, z_grid, colors = col[i])

    plt.title(title)
    plt.tight_layout()


# Load tempo and loudness features for a specified album
# tempo_loudness_album = load_tempo_loudness_for_album("/Users/nurupo/Desktop/dev/msd/blue_oyster/","Extraterrestrial Live")

# Load tempo and loudness features for a specified artist and album
#tempo_loudness_album = get_artist_scatter_feature("MNEMIC", "tempo", "loudness", True)
tempo_loudness_album = get_artist_scatter_feature("MNEMIC", "tempo", "loudness", True)

print(tempo_loudness_album)
if tempo_loudness_album.size == 0:
    print("No data to fit or plot.")
else:
    # Step 1: Fit a Gaussian Mixture Model (GMM) to the tempo and loudness features
    num_components = 1# Adjust the number of components (clusters) as needed
    gmm = GaussianMixture(n_components=num_components, covariance_type='full', random_state=42)
    gmm.fit(tempo_loudness_album)
    scores = gmm.score_samples(tempo_loudness_album)
    labels = gmm.predict(tempo_loudness_album)
    plot_contours(tempo_loudness_album, gmm.means_, gmm.covariances_,"AA")

    # Define threshold based on the 5th percentile of the scores to identify outliers

    thresh_ratio = 0.05
    thresh = np.quantile(scores, thresh_ratio)

    # Calculate the percentile q based on the threshold score
    # Find the score percentile that corresponds to the `thresh` score
    q = (1-thresh_ratio) * 100

    # Step 2: Plot the tempo and loudness data points with cluster assignments
    plt.figure(figsize=(8, 6))

    # Plot each cluster with a different color
    for i in range(num_components):
        cluster_points = tempo_loudness_album[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}', alpha=0.6)

    # Plot the means and boundary ellipses for each Gaussian component using Mahalanobis distance
    for i in range(num_components):
        mean = gmm.means_[i]
        cov = gmm.covariances_[i]

        # Calculate Mahalanobis distance for each point to determine the boundary
        inv_cov = np.linalg.inv(cov)
        max_distance = np.percentile(
            [mahalanobis(point, mean, inv_cov) for point in cluster_points], q
        )  # Use `q` percentile distance for boundary

        # Draw the ellipse based on Mahalanobis distance threshold
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        axis_length = 2 * np.sqrt(eigenvalues) * max_distance
        ellipse_angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

        ellipse = plt.matplotlib.patches.Ellipse(
            mean, axis_length[0], axis_length[1], angle=ellipse_angle,
            edgecolor='black', fc='none', lw=2, label=f'Mahalanobis Boundary {i + 1}'
        )
        plt.gca().add_patch(ellipse)

    # Annotate outliers: points with score below the threshold
    outliers = tempo_loudness_album[scores < thresh]
    plt.scatter(outliers[:, 0], outliers[:, 1], color='red', marker='^', label='Outliers', s=80)

    # Final plot adjustments
    plt.title("Gaussian Mixture Model Fit for Tempo and Loudness (With Score-Based Mahalanobis Boundary)")
    plt.xlabel("Tempo (BPM)")
    plt.ylabel("Loudness (dB)")
    plt.legend()
    plt.tight_layout()
    plt.show()
