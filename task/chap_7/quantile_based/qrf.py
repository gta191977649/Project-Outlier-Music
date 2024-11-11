import os
import feature.msd_getter as h5
import feature.msd_dataset as msd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import QuantileRegressor


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

    # Plotting the results
    plt.figure(figsize=(5, 4))

    # Plot non-outliers as blue dots with marker size 1
    plt.plot(data.index[~outliers], data[feature][~outliers], 'x', color='blue', markersize=8,
             label=f'{feature.capitalize()} Data Points')

    # Plot each quantile line
    for quantile in quantiles:
        q_label = f"{int(quantile * 100)}th"
        plt.plot(data.index, quantile_predictions[q_label], linestyle='--' if quantile != 0.5 else '-',
                 label=f'{q_label} Percentile')

    # Fill between 5th and 95th percentiles if those quantiles are specified
    if "5th" in quantile_predictions and "95th" in quantile_predictions:
        plt.fill_between(data.index, quantile_predictions["5th"], quantile_predictions["95th"], color='red', alpha=0.1)

    # Highlight outliers as red "+" with size 1
    if show_outliers:
        plt.plot(data.index[outliers], data[feature][outliers], 'r^', label='Outliers')

    # Customize plot appearance
    plt.xlabel("Song Index")
    plt.ylabel(f"{feature.capitalize()}")
    plt.title(f"Quantile Regression for {feature.capitalize()} with Outlier Detection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    rl = "Original Album Classics"
    releases = group_files_by_release("/Users/nurupo/Desktop/dev/msd/blue_oyster/")
    print(releases)
    collection = loadArtistCollection("/Users/nurupo/Desktop/dev/msd/blue_oyster/")
    x = []
    y = []
    for path in collection:
        song = msd.openSong(path)
        a = msd.getFeature(song, feature="tempo")
        b = msd.getFeature(song, feature="loudness")
        x.append(a)
        y.append(b)

    # Convert lists to DataFrame
    data = pd.DataFrame({'tempo': x, 'loudness': y})

    # Call function for "tempo" feature with 5th and 95th percentiles and outlier detection enabled
    fit_and_plot_quantile_regression(data, feature="tempo", quantiles=[0.05, 0.5, 0.95], show_outliers=True)

    # Call function for "loudness" feature with 5th and 95th percentiles and outlier detection enabled
    fit_and_plot_quantile_regression(data, feature="loudness", quantiles=[0.05, 0.5, 0.95], show_outliers=True)
