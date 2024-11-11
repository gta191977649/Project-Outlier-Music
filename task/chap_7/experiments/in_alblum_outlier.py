import os
import feature.msd_getter as h5
import feature.msd_dataset as msd
import matplotlib.pyplot as plt
import numpy as np
from sklearn_quantile import RandomForestQuantileRegressor
from scipy.spatial.distance import mahalanobis
from dtaidistance import dtw
from scipy.spatial.distance import cosine  # Import cosine distance from scipy

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
        # Model for the lower quantile
        lower_qrf = RandomForestQuantileRegressor(n_estimators=n_estimators, q=lower_quantile, random_state=0)
        lower_qrf.fit(np.arange(mfcc_data.shape[0]).reshape(-1, 1), mfcc_data[:, i])
        lower_qrf_models.append(lower_qrf)

        # Model for the upper quantile
        upper_qrf = RandomForestQuantileRegressor(n_estimators=n_estimators, q=upper_quantile, random_state=0)
        upper_qrf.fit(np.arange(mfcc_data.shape[0]).reshape(-1, 1), mfcc_data[:, i])
        upper_qrf_models.append(upper_qrf)

    return lower_qrf_models, upper_qrf_models


from dtaidistance import dtw  # Import DTW from dtaidistance library


def calculate_anomaly_score(mfcc, lower_qrf_models, upper_qrf_models, album_mean):
    quantile_deviation_sum = 0
    within_quantile_count = 0
    cosine_distance_sum = 0
    cosine_count = 0
    total_coefficients = len(mfcc)

    # Calculate slopes for the album mean
    album_slopes = album_mean[1:] - album_mean[:-1]

    # Calculate quantile-based scores
    for i in range(total_coefficients - 1):  # Iterate over each slope
        # Get quantile bounds for each coefficient
        lower = lower_qrf_models[i].predict(np.array([[total_coefficients]]))[0]
        upper = upper_qrf_models[i].predict(np.array([[total_coefficients]]))[0]
        lower_next = lower_qrf_models[i + 1].predict(np.array([[total_coefficients]]))[0]
        upper_next = upper_qrf_models[i + 1].predict(np.array([[total_coefficients]]))[0]

        # Check if both points are within bounds for Quantile Coverage Score
        if lower <= mfcc[i] <= upper:
            within_quantile_count += 1
            deviation = 0  # No deviation if both points are within bounds
            slope_cosine_distance = 0  # Perfect alignment, cosine distance = 0
        else:
            # Calculate deviation to the nearest bound for out-of-bounds points
            deviation = abs(mfcc[i] - (upper if mfcc[i] > upper else lower))
            quantile_deviation_sum += deviation


    # Quantile Coverage Score (unnormalized)
    print(within_quantile_count,total_coefficients)

    quantile_coverage_score = (within_quantile_count / total_coefficients) * 100
    #
    # # Quantile Deviation Score (unnormalized, modified to count 0 for in-range values)
    # quantile_deviation_score = 1 - (quantile_deviation_sum / total_coefficients)
    #
    # # Averaged Cosine Distance Score for gradient alignment (unnormalized)
    # cosine_distance_score = cosine_distance_sum / cosine_count if cosine_count > 0 else 0
    #
    # # Total Anomaly Score (sums the raw scores)
    # total_raw_score = quantile_coverage_score + quantile_deviation_score + (1 - cosine_distance_score)

    # Return both unnormalized and normalized scores
    return {
        "Quantile Coverage Score": quantile_coverage_score,
        # "Quantile Deviation Score": quantile_deviation_score,
        # "Averaged Cosine Distance Score (1 - cosine distance)":  cosine_distance_score,
        # "Total Anomaly Score": total_raw_score
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
    lower_qrf_models, upper_qrf_models = train_qrf_models(mfcc_coff, n_estimators=500, lower_quantile=0.05,
                                                          upper_quantile=0.95)
    album_mean = np.mean(mfcc_coff, axis=0)
    album_cov = np.cov(mfcc_coff, rowvar=False)

    for song_mfcc in mfcc_coff:
        anomaly_score = calculate_anomaly_score(song_mfcc, lower_qrf_models, upper_qrf_models,album_mean=album_mean)
        print(anomaly_score)



    # Visualize MFCCs with anomaly detection results
    plot_mfcc_with_anomalies(mfcc_coff, lower_qrf_models, upper_qrf_models,title=f"MFCC Outlier Detection for Release: {rl}")
