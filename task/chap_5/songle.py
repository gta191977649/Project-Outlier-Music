import librosa
import feature.extract as feature
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import seaborn as sns


def calculate_similarity(chroma):
    t, d = chroma.shape
    similarity = np.zeros((t, t))

    # Normalize chroma vectors
    chroma_norm = chroma / np.maximum(np.max(chroma, axis=1, keepdims=True), 1e-8)

    # Create progress bar
    total_iterations = sum(range(t + 1))  # Sum of 1 to t
    with tqdm(total=total_iterations, desc="Calculating Similarity", unit="comparison") as pbar:
        for tau in range(t):
            for l in range(tau + 1):
                diff = np.abs(chroma_norm[tau] - chroma_norm[tau - l])
                similarity[tau, l] = 1 - np.sum(diff) / np.sqrt(12)
                pbar.update(1)
    return similarity


def calculate_r_all(similarity):
    t, _ = similarity.shape
    r_all = np.zeros_like(similarity)

    with tqdm(total=t, desc="Calculating R_all", unit="lag") as pbar:
        for l in range(t):
            cumsum = 0
            for tau in range(l, t):
                cumsum += similarity[tau, l]
                r_all[tau, l] = cumsum / (tau - l + 1)
            pbar.update(1)

    return r_all.T


def apply_moving_average(r_all, window_size=5):
    """Apply a moving average filter to R_all."""
    t, _ = r_all.shape
    r_all_smoothed = np.zeros_like(r_all)

    with tqdm(total=t, desc="Applying Moving Average", unit="column") as pbar:
        for l in range(t):
            r_all_smoothed[:, l] = np.convolve(r_all[:, l],
                                               np.ones(window_size) / window_size,
                                               mode='same')
            pbar.update(1)

    return r_all_smoothed


def optimize_threshold(values):
    sorted_values = np.sort(values)
    total_sum = np.sum(values)
    total_count = len(values)

    max_variance = 0
    optimal_threshold = 0

    cumulative_sum = 0
    for i, threshold in enumerate(sorted_values):
        class1_count = i + 1
        class2_count = total_count - class1_count

        if class1_count == 0 or class2_count == 0:
            continue

        cumulative_sum += threshold
        class1_mean = cumulative_sum / class1_count
        class2_mean = (total_sum - cumulative_sum) / class2_count

        class_weight1 = class1_count / total_count
        class_weight2 = class2_count / total_count

        between_class_variance = class_weight1 * class_weight2 * (class1_mean - class2_mean) ** 2

        if between_class_variance > max_variance:
            max_variance = between_class_variance
            optimal_threshold = threshold

    return optimal_threshold


def list_repeated_sections(r_all, min_duration=5, frame_rate=1 / 0.08, threshold=None):
    """
    List repeated sections based on the corrected R_all matrix.

    :param r_all: The corrected R_all matrix (transposed, with time on x-axis and lag on y-axis)
    :param min_duration: Minimum duration of a repeated section in seconds
    :param frame_rate: Number of frames per second in the chroma features
    :param threshold: User-defined threshold. If None, threshold will be optimized for each time column
    :return: List of repeated sections
    """
    t, l = r_all.shape  # t is now the number of time frames, l is the number of lags
    min_frames = int(min_duration * frame_rate)
    repeated_sections = []

    with tqdm(total=t, desc="Finding Repeated Sections", unit="time") as pbar:
        for time in range(t):
            # Use user-defined threshold or optimize
            if threshold is None:
                optimal_threshold = optimize_threshold(r_all[:, time])
            else:
                optimal_threshold = threshold

            # Find peaks in R_all for each time column
            peaks, _ = find_peaks(r_all[:, time], height=optimal_threshold, distance=min_frames)

            for peak in peaks:
                # Find the start and end of the line segment
                start = time
                end = time

                # Look backwards
                while start > 0 and r_all[peak, start - 1] > optimal_threshold:
                    start -= 1

                # Look forwards
                while end < t - 1 and r_all[peak, end + 1] > optimal_threshold:
                    end += 1

                duration = (end - start + 1) / frame_rate
                if duration >= min_duration:
                    repeated_sections.append({
                        'start': start / frame_rate,
                        'end': (end + 1) / frame_rate,  # +1 because end is inclusive
                        'lag': peak / frame_rate,
                        'r_value': r_all[peak, time]
                    })
            pbar.update(1)

    # Sort repeated sections by r_value
    repeated_sections.sort(key=lambda x: x['r_value'], reverse=True)

    return repeated_sections


def plot_similarity_heatmap(similarity):
    """
    Plot the full similarity matrix as a heatmap using Seaborn with a color scale from 0 to 1.
    """
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))

    # Create the heatmap using seaborn
    sns.heatmap(similarity,
                cmap='viridis',
                square=True,
                cbar_kws={'label': 'Similarity'},
                vmin=0,  # Set the minimum value for the colorscale
                vmax=1)  # Set the maximum value for the colorscale

    plt.title('Full Similarity Matrix Heatmap')
    plt.xlabel('Lag')
    plt.ylabel('Time')

    # Invert the y-axis to match the original matrix orientation
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()


def plot_r_all(r_all):
    """
        Plot R_all for all frames and visualize the optimal thresholds and peaks.

        :param r_all: The R_all matrix
        """
    num_frames, num_lags = r_all.shape

    # Prepare data structures
    optimal_thresholds = np.zeros(num_frames)
    all_peaks = []

    # Calculate optimal thresholds and find peaks for each frame
    print("Calculating optimal thresholds and finding peaks...")
    for t in tqdm(range(num_frames)):
        r_all_column = r_all[:, t]
        optimal_thresholds[t] = optimize_threshold(r_all_column)
        peaks, _ = find_peaks(r_all_column, height=optimal_thresholds[t], distance=5)
        all_peaks.append(peaks)

    # Create the plot
    plt.figure(figsize=(20, 10))

    # Plot R_all as a heatmap
    plt.imshow(r_all.T, aspect='auto', origin='lower', cmap='viridis', extent=[0, num_lags, 0, num_frames])
    plt.colorbar(label='R_all Value')

    # Plot optimal thresholds
    plt.plot(np.arange(num_frames), optimal_thresholds, color='red', linewidth=2, label='Optimal Threshold')

    # Plot peaks
    for t, peaks in enumerate(all_peaks):
        if len(peaks) > 0:
            plt.scatter(peaks, [t] * len(peaks), color='white', s=10, alpha=0.5)

    plt.title('R_all Matrix with Optimal Thresholds and Peaks')
    plt.xlabel('Lag (frames)')
    plt.ylabel('Time (frames)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Average Optimal Threshold: {np.mean(optimal_thresholds):.4f}")
    print(f"Minimum Optimal Threshold: {np.min(optimal_thresholds):.4f}")
    print(f"Maximum Optimal Threshold: {np.max(optimal_thresholds):.4f}")
    print(f"Total number of peaks detected: {sum(len(peaks) for peaks in all_peaks)}")


if __name__ == '__main__':

    SONG_FILE = "/Users/nurupo/Desktop/dev/audio/bic_camera.mp3"
    # 1. Extracting acoustic features (chroma)
    chroma = feature.extract_feature(SONG_FILE, "chroma_stft")

    print("Step 1: Calculating Similarity")
    similarity = calculate_similarity(chroma.T)

    print("\nStep 2: Calculating R_all")
    r_all = calculate_r_all(similarity)

    plot_r_all(r_all)
    print("\nStep 3: Applying Moving Average")
    r_all_smoothed = apply_moving_average(r_all)


    plot_similarity_heatmap(similarity)
    plot_similarity_heatmap(r_all_smoothed)
    print("\nStep 4: Finding Repeated Sections")
    repeated_sections = list_repeated_sections(r_all_smoothed)

    print("\nTop 10 repeated sections:")
    for i, section in enumerate(repeated_sections[:10]):
        print(f"{i + 1}. Start: {section['start']:.2f}s, End: {section['end']:.2f}s, "
              f"Duration: {section['end'] - section['start']:.2f}s, "
              f"Lag: {section['lag']:.2f}s, R-value: {section['r_value']:.4f}")

    print("done")