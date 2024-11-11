import numpy as np
from sklearn.neighbors import NearestNeighbors
# Sample data (e.g., MFCC features for 5 songs, with 12 MFCC coefficients each)
samples = np.array([
    [1.2, 0.5, -0.3, 1.5, 0.9, -1.1, 0.8, -0.6, 1.0, -0.4, 0.2, -0.9],
    [0.9, 0.3, -0.2, 1.3, 0.7, -1.0, 0.6, -0.5, 0.9, -0.3, 0.1, -0.8],
    [1.1, 0.4, -0.1, 1.4, 0.8, -1.2, 0.7, -0.4, 0.8, -0.5, 0.3, -0.7],
    [1.0, 0.6, -0.4, 1.2, 0.6, -1.3, 0.9, -0.7, 1.1, -0.6, 0.5, -0.5],
    [1.3, 0.7, -0.5, 1.6, 1.0, -0.9, 1.0, -0.3, 1.2, -0.2, 0.6, -1.0]
])

# Define the lower and upper bounds for each feature (e.g., based on quantiles or empirical ranges)
lower_bounds = np.array([0.8, 0.2, -0.5, 1.0, 0.5, -1.5, 0.5, -0.8, 0.7, -0.7, 0.0, -1.2])
upper_bounds = np.array([1.4, 0.8, 0.0, 1.7, 1.2, -0.5, 1.2, -0.2, 1.3, -0.1, 0.6, -0.3])


def sparsity_within_bounds(samples, lower_bounds, upper_bounds):
    total_features = samples.shape[1]
    within_bounds_counts = 0
    in_bounds_distances = []

    # Iterate over each sample and check if features are within bounds
    for sample in samples:
        in_bounds = np.logical_and(sample >= lower_bounds, sample <= upper_bounds)
        within_bounds_counts += np.sum(in_bounds)

        # Filter only features that are within bounds for density measurement
        in_bound_features = sample[in_bounds]
        if len(in_bound_features) > 1:
            nbrs = NearestNeighbors(n_neighbors=2).fit(in_bound_features.reshape(-1, 1))
            distances, _ = nbrs.kneighbors(in_bound_features.reshape(-1, 1))
            # Ignore self-distance and calculate mean distance for density within bounds
            in_bounds_distances.append(np.mean(distances[:, 1]))

    # 1. Coverage Percentage: % of features within bounds
    coverage_percentage = (within_bounds_counts / (samples.shape[0] * total_features)) * 100

    # 2. Average Distance for Density within bounds
    avg_in_bounds_distance = np.mean(in_bounds_distances) if in_bounds_distances else None

    # 3. Variability of Distances within bounds
    distance_variation = (np.std(in_bounds_distances) / np.mean(in_bounds_distances)
                          if in_bounds_distances and np.mean(in_bounds_distances) > 0 else None)

    # Return the sparsity measures
    return {
        "Coverage Percentage": coverage_percentage,
        "Average In-Bounds Distance": avg_in_bounds_distance,
        "In-Bounds Distance Variability": distance_variation
    }
# Run the sparsity_within_bounds function
sparsity_metrics = sparsity_within_bounds(samples, lower_bounds, upper_bounds)

# Print the results
print("Sparsity Metrics:")
print(f"Coverage Percentage: {sparsity_metrics['Coverage Percentage']:.2f}%")
print(f"Average In-Bounds Distance: {sparsity_metrics['Average In-Bounds Distance']:.4f}")
print(f"In-Bounds Distance Variability: {sparsity_metrics['In-Bounds Distance Variability']:.4f}")
