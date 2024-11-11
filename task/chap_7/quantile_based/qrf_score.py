import os
import feature.msd_getter as h5
import feature.msd_dataset as msd
import matplotlib.pyplot as plt
import numpy as np
from sklearn_quantile import RandomForestQuantileRegressor
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


def train_qrf_models(mfcc_data, n_estimators=100, lower_quantile=0.05, upper_quantile=0.95):
    lower_qrf_models = []
    upper_qrf_models = []

    for i in range(mfcc_data.shape[1]):  # Each MFCC coefficient
        # Model for the lower quantile (e.g., 5th percentile as 0.05)
        lower_qrf = RandomForestQuantileRegressor(n_estimators=n_estimators, q=lower_quantile, random_state=0)
        lower_qrf.fit(np.arange(mfcc_data.shape[0]).reshape(-1, 1), mfcc_data[:, i])
        lower_qrf_models.append(lower_qrf)

        # Model for the upper quantile (e.g., 95th percentile as 0.95)
        upper_qrf = RandomForestQuantileRegressor(n_estimators=n_estimators, q=upper_quantile, random_state=0)
        upper_qrf.fit(np.arange(mfcc_data.shape[0]).reshape(-1, 1), mfcc_data[:, i])
        upper_qrf_models.append(upper_qrf)

    return lower_qrf_models, upper_qrf_models



def calculate_anomaly_score(mfcc, lower_qrf_models, upper_qrf_models, album_mean, album_cov):
    quantile_deviation_sum = 0
    within_quantile_count = 0
    total_coefficients = len(mfcc)
    # Calculate quantile-based scores
    for i, (lower_qrf, upper_qrf) in enumerate(zip(lower_qrf_models, upper_qrf_models)):  # Each MFCC coefficient
        lower = lower_qrf.predict(np.array([[total_coefficients]]))[0]
        upper = upper_qrf.predict(np.array([[total_coefficients]]))[0]

        # Check if within quantile range for Quantile Coverage Score
        if lower <= mfcc[i] <= upper:
            within_quantile_count += 1
        # Calculate deviation for Quantile Deviation Score
        quantile_deviation_sum += abs(mfcc[i] - (upper if mfcc[i] > upper else lower))

    # Quantile Coverage Fit Score
    quantile_coverage_score = (within_quantile_count / total_coefficients) * 100

    # Quantile Deviation Score (normalized by total coefficients)
    quantile_deviation_score = 1 - (quantile_deviation_sum / total_coefficients)

    # Mahalanobis Distance Score
    # Using scipy.spatial.distance.mahalanobis
    # print(mfcc.shape)
    # print(album_mean.shape)
    # print(album_cov.shape)
    album_cov_inv = np.linalg.pinv(album_cov)
    mahalanobis_distance = mahalanobis(mfcc, album_mean, album_cov_inv)
    mahalanobis_score = np.exp(-mahalanobis_distance)

    # Return all three scores as a dictionary
    return {
        "Quantile Coverage Score": quantile_coverage_score,
        "Quantile Deviation Score": quantile_deviation_score,
        "Mahalanobis Distance Score": mahalanobis_score
    }



def plot_mfcc_with_anomalies(mfcc_coff, lower_qrf_models, upper_qrf_models, title):
    mean_mfcc = np.mean(mfcc_coff, axis=0)
    std_mfcc = np.std(mfcc_coff, axis=0)

    # Calculate the predicted lower and upper quantiles for each coefficient
    lower_bounds = np.array([lower_qrf.predict(np.array([[len(mfcc_coff)]]))[0] for lower_qrf in lower_qrf_models])
    upper_bounds = np.array([upper_qrf.predict(np.array([[len(mfcc_coff)]]))[0] for upper_qrf in upper_qrf_models])


    # Plotting with error bars for mean and std deviatio n
    plt.figure(figsize=(8, 6))
    plt.style.use('classic')
    # Plot all sample
    for mfcc in mfcc_coff:
        plt.plot(mfcc,alpha=0.35)

    plt.errorbar(range(mean_mfcc.shape[0]), mean_mfcc, yerr=std_mfcc, fmt='o', color="black", ecolor='black', capsize=5)
    plt.plot(range(mean_mfcc.shape[0]), mean_mfcc, linestyle='--', color='black',
             label="Mean MFCC")  # Line to connect points

    # Red shaded area for predicted quantile range
    plt.fill_between(range(len(lower_bounds)), lower_bounds, upper_bounds, color='red', alpha=0.2,
                     label="Predicted Range (5th-95th Percentile)")


    plt.xlabel("Coefficient Index")
    plt.ylabel("MFCC Coefficient Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    #rl = 'Secret Treaties'
    rl = "Secret Treaties"
    #rl = "Extraterrestrial Live"
    releases = group_files_by_release("/Users/nurupo/Desktop/dev/msd/blue_oyster/")
    print(releases)

    mfcc_coff = []
    for path in releases[rl]:
        song = msd.openSong(path)
        mfcc = msd.getFeature(song, feature="mfcc")
        mfcc_coff.append(mfcc)  # Directly append each MFCC

    mfcc_coff = np.array(mfcc_coff)  # Shape: (num_songs, num_coefficients)
    # Train QRF models on each MFCC coefficient for lower and upper quantiles
    lower_qrf_models, upper_qrf_models = train_qrf_models(mfcc_coff, n_estimators=100, lower_quantile=0.05,
                                                          upper_quantile=0.95)
    album_mean = np.mean(mfcc_coff, axis=0)
    album_cov = np.cov(mfcc_coff, rowvar=False)

    for song_mfcc in mfcc_coff:
        anomaly_score = calculate_anomaly_score(song_mfcc, lower_qrf_models, upper_qrf_models,album_mean=album_mean,album_cov=album_cov)
        print(anomaly_score)



    # Visualize MFCCs with anomaly detection results
    plot_mfcc_with_anomalies(mfcc_coff, lower_qrf_models, upper_qrf_models,title=f"MFCC Outlier Detection for Release: {rl}")
