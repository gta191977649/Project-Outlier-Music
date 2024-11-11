import os
import feature.msd_getter as h5
import feature.msd_dataset as msd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import math
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
            collection.append(full_path)
    return collection

# def train_quantile_regressor(data,feature,quantiles=0.5,alpha=0.1):
#     X = np.arange(len(data)).reshape(-1, 1)  # Song index as independent variable
#     qr = QuantileRegressor(quantile=quantiles, alpha=alpha)
#     qr.fit(X, data[feature])
#     predition = qr.predict(X)


def sparsity_within_bounds(samples, lower_bounds, upper_bounds):
   return {
       "bounds_range" : abs(lower_bounds[0] - upper_bounds[0])
   }
def fit_and_plot_quantile_regression(data, feature, quantiles=[0.05, 0.5, 0.95], alpha=0.1, show_outliers=False):
    """
    Fits and plots quantile regression for a specified feature, with an option to highlight outliers.

    Parameters:
    - data: DataFrame containing the features for each song.
    - feature: The name of the feature column to perform quantile regression on.
    - quantiles: List of quantiles to fit (e.g., [0.05, 0.5, 0.95]).
    - alpha: Regularization parameter for QuantileRegressor (default is 0.1).
    - show_outliers: If True, highlights data points outside the quantile range in red.
    """
    # Prepare the input data (song indices as X)
    X = np.arange(len(data)).reshape(-1, 1)  # Song index as independent variable

    # Fit QuantileRegressor for each specified quantile
    quantile_predictions = {}
    for quantile in quantiles:
        qr = QuantileRegressor(quantile=quantile, alpha=alpha)
        qr.fit(X, data[feature])
        quantile_predictions[f'{int(quantile * 100)}th'] = qr.predict(X)

    # Identify outliers if show_outliers is True
    if show_outliers and "5th" in quantile_predictions and "95th" in quantile_predictions:
        lower_bound = quantile_predictions["5th"]
        upper_bound = quantile_predictions["95th"]
        outliers = (data[feature] < lower_bound) | (data[feature] > upper_bound)
    else:
        outliers = np.array([False] * len(data))  # No outliers

    #print(data[feature].values)
    sparsity_metrics = sparsity_within_bounds(data[feature].values, quantile_predictions["5th"], quantile_predictions["95th"])
    print(sparsity_metrics)
    # Plotting the results
    plt.figure(figsize=(10, 8))

    # Plot non-outliers as blue dots with marker size 1
    plt.plot(data.index[~outliers], data[feature][~outliers], 'o', color='blue', markersize=8,
             label=f'{feature.capitalize()} Data Points')

    # Plot each quantile line
    for quantile in quantiles:
        q_label = f"{int(quantile * 100)}th"
        plt.plot(data.index, quantile_predictions[q_label],color='black' ,linestyle='--' if quantile != 0.5 else '-',
                 label=f'{q_label} Percentile', markersize=10)

    # Fill between 5th and 95th percentiles if those quantiles are specified
    if "5th" in quantile_predictions and "95th" in quantile_predictions:
        plt.fill_between(data.index, quantile_predictions["5th"], quantile_predictions["95th"], color='red', alpha=0.1)

    # Highlight outliers as red "+" with size 1
    if show_outliers:
        plt.plot(data.index[outliers], data[feature][outliers], 'rx', markersize=10, label='Outliers')

    # Customize plot appearance
    plt.xlabel("Song Index")
    plt.ylabel(f"{feature.capitalize()}")
    plt.title(f"Quantile Regression for {feature.capitalize()} with Outlier Detection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_artist_feature(artist, x="tempo",normalize=False):
    songs = msd.loadMSDCsvData("/Users/nurupo/Desktop/dev/msd/song_v2.csv")
    df = pd.DataFrame(songs)
    df_artist = df[df['artist'] == artist]
    X = df_artist[x]

    if normalize:
        # Reshape X and Y to 2D arrays for the scaler
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X.values.reshape(-1, 1)).flatten()

    features = X
    return np.array(features)

def get_album_feature(artist,album, x="tempo",normalize=False):
    songs = msd.loadMSDCsvData("/Users/nurupo/Desktop/dev/msd/song_v2.csv")
    df = pd.DataFrame(songs)
    df_artist = df[df['artist'] == artist]
    X = df_artist[x]

    if normalize:
        # Reshape X and Y to 2D arrays for the scaler
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X.values.reshape(-1, 1)).flatten()

    features = X
    return np.array(features)




if __name__ == '__main__':

    artist = "Blue Oyster Cult"
    x = get_artist_feature(artist,"tempo", normalize=False)
    y = get_artist_feature(artist,"loudness", normalize=False)

    # Convert lists to DataFrame
    data = pd.DataFrame({'tempo': x, 'loudness': y})

    # Call function for "tempo" feature with 5th and 95th percentiles and outlier detection enabled
    fit_and_plot_quantile_regression(data, feature="tempo", quantiles=[0.05, 0.5, 0.95], show_outliers=True)

    # Call function for "loudness" feature with 5th and 95th percentiles and outlier detection enabled
    fit_and_plot_quantile_regression(data, feature="loudness", quantiles=[0.05, 0.5, 0.95], show_outliers=True)
