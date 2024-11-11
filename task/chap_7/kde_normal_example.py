import numpy as np
import matplotlib.pyplot as plt

# Define the multivariate Gaussian kernel function
def gaussian_kernel(u):
    # u is the standardized distance between data points and evaluation points
    # d is the number of dimensions (features) of the data
    d = u.shape[1]
    # normalization constant for the Gaussian kernel in d dimensions
    normalization = (2 * np.pi) ** (-d / 2)
    # compute the Gaussian function
    return normalization * np.exp(-0.5 * np.sum(u**2, axis=1))

# Multidimensional KDE implementation
def kde_2d(x, data, bandwidth):
    # n: number of data points
    n = data.shape[0]
    # d: dimensionality of the data
    d = x.shape[1]
    # Initialize an array to store the density estimate at each point in x
    estimate = np.zeros(x.shape[0])
    # Loop over each data point to compute its contribution to the density estimate
    for i in range(n):
        # Calculate the standardized distance u using the bandwidth
        u = (x - data[i]) / bandwidth
        # Accumulate the Gaussian kernel contributions from each data point
        estimate += gaussian_kernel(u)
    # Normalize the density estimate by the number of points and the bandwidth raised to the dimensionality
    estimate /= (n * (bandwidth ** d))
    return estimate

# Generate sample data with 2 features and 2 centers
np.random.seed(42)  # For reproducibility
data_center1 = np.random.normal(0, 1, (500, 2))
data_center2 = np.random.normal(5, 1, (500, 2))

# Manually add a few outliers
outliers = np.array([[10, 10], [11, 11], [-3, 10], [7, -5]])
data = np.vstack((data_center1, data_center2, outliers))

# Create a grid for KDE evaluation and visualization
x = np.linspace(min(data[:, 0]) - 2, max(data[:, 0]) + 2, 100)
y = np.linspace(min(data[:, 1]) - 2, max(data[:, 1]) + 2, 100)
X, Y = np.meshgrid(x, y)
xy = np.vstack([X.ravel(), Y.ravel()]).T

# Compute KDE on the grid
bandwidth = 0.5  # Bandwidth affects the smoothness of the KDE
pdf = kde_2d(xy, data, bandwidth).reshape(X.shape)

# Detect outliers: points with very low probability density
threshold = 0.001  # Threshold for density to consider a point an outlier
outlier_mask = kde_2d(data, data, bandwidth) < threshold
detected_outliers = data[outlier_mask]

# Plotting the results
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, pdf, levels=50, cmap='Blues')  # Density plot
plt.colorbar(label='Density')
plt.scatter(data[:, 0], data[:, 1], s=5, color='k', alpha=0.5, label='Data Points')  # Data points
plt.scatter(detected_outliers[:, 0], detected_outliers[:, 1], s=50, color='red', label='Detected Outliers')  # Outliers
plt.title('2D KDE and Outlier Detection with Two Centers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()