import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, multivariate_normal


def generate_data(n_data, means, covariances, weights):
    """creates a list of data points"""
    n_clusters, n_features = means.shape

    data = np.zeros((n_data, n_features))
    for i in range(n_data):
        # pick a cluster id and create data from this cluster
        k = np.random.choice(n_clusters, size=1, p=weights)[0]
        x = np.random.multivariate_normal(means[k], covariances[k])
        data[i] = x

    return data

def plot_contours(data, means, covs, title):
    """visualize the gaussian components over the data"""
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], 'ko')

    delta = 0.025
    k = means.shape[0]
    x = np.arange(-2.0, 7.0, delta)
    y = np.arange(-2.0, 7.0, delta)
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
    plt.show()


if __name__ == '__main__':
    init_means = np.array([
        [5, 0],
        [1, 1],
        [0, 5]
    ])

    init_covariances = np.array([
        [[.5, 0.], [0, .5]],
        [[.92, .38], [.38, .91]],
        [[.5, 0.], [0, .5]]
    ])

    init_weights = [1 / 4, 1 / 2, 1 / 4]

    # generate data
    np.random.seed(4)
    X = generate_data(100, init_means, init_covariances, init_weights)

    gmm = GaussianMixture(n_components=3, covariance_type='full',
                          max_iter=600, random_state=3)
    gmm.fit(X)

    print('converged or not: ', gmm.converged_)
    plot_contours(X, gmm.means_, gmm.covariances_, 'Final clusters')