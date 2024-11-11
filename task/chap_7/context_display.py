from feature.dataset import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

import numpy as np
def loadMusic4AllCsvData(path,mode):
    songs = []
    csv = pd.read_csv(path)
    for idx,item in csv.iterrows():
        if item['mode'] == mode:
            songs.append(item.to_dict())
    return songs

def loadMSDCsvData(path):
    songs = []
    csv = pd.read_csv(path)
    for idx,item in csv.iterrows():
        songs.append(item.to_dict())
    return songs


def plot(songs):
    # Convert the list of dictionaries into a DataFrame for easier grouping and plotting
    df = pd.DataFrame(songs)

    # Create a scatter plot with different colors for each artist
    #plt.figure(figsize=(10, 7))

    # Use seaborn to color by "artist" and automatically generate a color palette
    sns.scatterplot(data=df, x='x', y='y', hue='artist', color="blue", alpha=0.7)

    # Add plot details
    plt.title('Scatter Plot of Tempo vs Energy (Grouped by Artist)')
    plt.xlabel('Tempo')
    plt.ylabel('Energy')
    plt.grid(True)

    # Display the legend outside the plot for better visibility
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()

def plot_artist(songs, artist):
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(songs)

    # Filter the DataFrame to only include the specified artist
    df_artist = df[df['artist'] == artist]

    # Check if the artist has any songs in the dataset
    if df_artist.empty:
        print(f"No songs found for artist: {artist}")
        return

    # Create a scatter plot for the specified artist
    #plt.figure(figsize=(10, 7))

    # Plot tempo vs danceability for the artist's songs
    sns.scatterplot(data=df_artist, x='tempo', y='loudness', color="blue",alpha=0.7)

    # Add plot details
    plt.title(f'Scatter Plot of Tempo vs Danceability for {artist}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

    plt.show()


def get_artist_scatter_feature(artist, x="tempo", y="loudness", normalize=False):
    df = pd.DataFrame(songs)
    df_artist = df[df['artist'] == artist]

    X = df_artist[x]
    Y = df_artist[y]

    if normalize:
        # Reshape X and Y to 2D arrays for the scaler
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X.values.reshape(-1, 1)).flatten()
        Y = scaler.fit_transform(Y.values.reshape(-1, 1)).flatten()

    return X, Y


def cluster_and_plot(x, y, method="kmeans", n_clusters=2, eps=0.5, min_samples=5):
    x_name = "x" if isinstance(x, (np.ndarray, list)) else x.name
    y_name = "y" if isinstance(y, (np.ndarray, list)) else y.name

    data = pd.DataFrame({x_name: x, y_name: y})

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=0)
    elif method == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=0)
    else:
        raise ValueError("Unknown clustering method. Choose from 'kmeans', 'dbscan', 'agglomerative', or 'gmm'.")

    # For GMM, we use the predict method rather than fit_predict
    if method == "gmm":
        model.fit(data)
        data['cluster'] = model.predict(data)
    else:
        data['cluster'] = model.fit_predict(data)

    sns.scatterplot(data=data, x=x_name, y=y_name, hue='cluster', palette="tab20", legend="full")
    plt.title(f"Clustering on {x_name} and {y_name} using {method.capitalize()}")
    plt.show()


def find_optimal_clusters(x, y, method="kmeans", n_single_element_clusters=1, start_n_clusters=2, end_n_clusters=30):
    x_name = x.name if isinstance(x, pd.Series) else "x"
    y_name = y.name if isinstance(y, pd.Series) else "y"
    data = pd.DataFrame({x_name: x, y_name: y})

    for n_clusters in range(start_n_clusters, end_n_clusters + 1):
        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=0)
            labels = model.fit_predict(data)
            inertia = model.inertia_
            description = f"Inertia: {inertia:.2f}"
        elif method == "gmm":
            model = GaussianMixture(n_components=n_clusters, random_state=0)
            model.fit(data)
            labels = model.predict(data)
            score = model.score(data)
            description = f"Log likelihood: {score:.2f}"
        else:
            raise ValueError("Unknown method. Choose from 'kmeans' or 'gmm'.")

        print(f"Clustering with {n_clusters} clusters, {description}")

        cluster_counts = pd.Series(labels).value_counts()
        single_element_clusters = (cluster_counts == 1).sum()

        if single_element_clusters == n_single_element_clusters:
            print(f"{n_single_element_clusters} clusters with 1 element found at {n_clusters} clusters. Optimal number of clusters is {n_clusters}")
            return n_clusters

    print(f"No solution found with {n_single_element_clusters} clusters with 1 element within range. Returning the end_n_clusters: {end_n_clusters}")
    return end_n_clusters

def one_class_outlier_detection(x, y, nu=0.5, kernel="linear", gamma='auto'):
    x_name = "x" if isinstance(x, (np.ndarray, list)) else x.name
    y_name = "y" if isinstance(y, (np.ndarray, list)) else y.name

    data = pd.DataFrame({x_name: x, y_name: y})

    # Reshape the data for the model
    X_data = data[[x_name, y_name]].values

    # Fit the One-Class SVM model
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X_data)

    # Predict outliers (-1 for outliers, 1 for inliers)
    data['prediction'] = model.predict(X_data)

    # Separate inliers and outliers
    inliers = data[data['prediction'] == 1]
    outliers = data[data['prediction'] == -1]

    # Plotting
    plt.figure(figsize=(10, 7))
    plt.scatter(inliers[x_name], inliers[y_name], color='blue', label='Inliers')
    plt.scatter(outliers[x_name], outliers[y_name], color='red', label='Outliers')
    plt.title('One-Class SVM Outlier Detection')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()
    plt.grid(True)
    plt.show()


def isolation_forest_outlier_detection(x, y, contamination=0.1, n_estimators=100, max_samples='auto', random_state=42,
                                       threshold=None):
    x_name = "x" if isinstance(x, (np.ndarray, list)) else x.name
    y_name = "y" if isinstance(y, (np.ndarray, list)) else y.name

    data = pd.DataFrame({x_name: x, y_name: y})

    # Prepare the data
    X_data = data[[x_name, y_name]].values

    # Fit the Isolation Forest model
    model = IsolationForest(contamination=contamination, n_estimators=n_estimators,
                            max_samples=max_samples, random_state=random_state)
    model.fit(X_data)

    # Compute anomaly scores (the lower, the more abnormal)
    scores = model.decision_function(X_data)
    data['anomaly_score'] = scores

    # If a threshold is provided, use it to classify outliers
    if threshold is not None:
        # Points with decision_function <= threshold are considered outliers
        data['prediction'] = np.where(data['anomaly_score'] <= threshold, -1, 1)
    else:
        # Use model's default threshold based on contamination
        data['prediction'] = model.predict(X_data)

    # Separate inliers and outliers
    inliers = data[data['prediction'] == 1]
    outliers = data[data['prediction'] == -1]

    # Print anomaly scores
    print("Anomaly Scores:")
    print(data[['anomaly_score']])

    # Plotting
    plt.figure(figsize=(10, 7))
    plt.scatter(inliers[x_name], inliers[y_name], color='blue', label='Inliers')
    plt.scatter(outliers[x_name], outliers[y_name], color='red', label='Outliers')
    plt.title('Isolation Forest Outlier Detection')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()
    plt.grid(True)
    plt.show()

    return data  # Return data for further analysis

def plot_anomaly_scores(data):
    plt.figure(figsize=(10, 7))
    sns.histplot(data['anomaly_score'], kde=True)
    plt.title('Anomaly Score Distribution')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
if __name__ == '__main__':
    songs = loadMSDCsvData("/Users/nurupo/Desktop/dev/msd/song_v2.csv")
    #X,Y = get_artist_scatter_feature("Blue Oyster Cult","tempo","loudness")
    #X,Y = get_artist_scatter_feature("Frankie Valli","tempo","loudness")
    #X,Y = get_artist_scatter_feature("The Mission","tempo","loudness")
    X,Y = get_artist_scatter_feature("Daedelus","tempo","loudness") #dance


    #n = find_optimal_clusters(X,Y,n_single_element_clusters=4,method="gmm")
    #cluster_and_plot(X,Y,method="kmeans",n_clusters=2)

    # Apply Isolation Forest outlier detection and get the data with anomaly scores
    data = isolation_forest_outlier_detection(X, Y, contamination=0.1, n_estimators=100, max_samples='auto',
                                              random_state=42)

    # Plot the anomaly score distribution
    plot_anomaly_scores(data)

    # Decide on a custom threshold (e.g., mean minus one standard deviation)
    threshold = data['anomaly_score'].mean() - data['anomaly_score'].std()
    print(f"Using custom threshold: {threshold}")

    # Apply the threshold
    data = isolation_forest_outlier_detection(X, Y, contamination=0.1, n_estimators=100,
                                              max_samples='auto', random_state=42, threshold=-0.04)