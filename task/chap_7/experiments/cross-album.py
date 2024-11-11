import feature.msd_dataset as msd
from sklearn.linear_model import QuantileRegressor
from sklearn_quantile import RandomForestQuantileRegressor
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from scipy.stats import norm, multivariate_normal
import matplotlib
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
import math
import seaborn as sns
import re



class RegressionContextModel:
    def __init__(self,data,quantiles=[0.05, 0.5, 0.95],alpha = 0.1,norm_max=178,norm_min=24):
        self.quantile_predictions = {}
        x = np.arange(len(data)).reshape(-1, 1)
        y = data
        for quantile in quantiles:
            qr = QuantileRegressor(quantile=quantile, alpha=alpha)
            qr.fit(x,y)
            self.quantile_predictions[f'{int(quantile * 100)}th'] = qr.predict(x)

        # compute context boundary
        self.lower_bound = self.quantile_predictions["5th"]
        self.upper_bound = self.quantile_predictions["95th"]
        self.mean = np.mean(self.quantile_predictions["50th"])
        # bounds range (in normalized form)
        # norm_max = 178 # Prestissimo (fatest)
        # norm_min = 24 # Larghissimo (slowest)
        self.bound_range = np.abs(np.mean(self.upper_bound)) - np.abs(np.mean(self.lower_bound)) / (norm_max - norm_min)
        self.bound_range = np.abs(self.bound_range)
        print(self.bound_range)
        # list outliers
        self.outliers = (data < self.lower_bound) | (data > self.upper_bound)
    def get_outliers(self):
        return self.outliers

class GaussianContextModel:
    def __init__(self,data,n_components=1,threshold=0.035,covariance_type='full', random_state=42):
        # self.x = x
        # self.y = y
        #self.data = np.array(list(zip(x,y)))
        self.data = data

        # init prediction model
        self.gmm = GaussianMixture(n_components=n_components,covariance_type=covariance_type,random_state=random_state,max_iter = 600)
        self.gmm.fit(self.data)
        self.centroids = self.gmm.means_ # centroids is array
        self.covariances = self.gmm.covariances_
        self.scores = self.gmm.score_samples(self.data)
        self.labels = self.gmm.predict(self.data)
        self.threshold = threshold
        # Define threshold based on the 5th percentile of the scores to identify outliers
        thresh = np.quantile(self.scores, self.threshold)

        self.outliers = self.scores < thresh

    def get_outliers(self):
        return self.outliers

    def plot(self):
        plt.figure(figsize=(5, 4))
        k = self.centroids.shape[0]
        for i in range(k):
            mean = self.centroids[i]
            cov = self.covariances[i]
            cluster_points = self.data[self.labels == i]
            q = (1 - self.threshold) * 100
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
                mean,
                axis_length[0],
                axis_length[1],
                angle=ellipse_angle,
                edgecolor='black',
                facecolor=(1, 0, 0, 0.1),
                #alpha=0.2,  # Set transparency (0 is transparent, 1 is solid)
                lw=1,  # Line width
                label=f'Context Area{i + 1}'
            )
            plt.gca().add_patch(ellipse)

        # plot sample
        plt.scatter(x=self.data[~self.outliers, 0], y=self.data[~self.outliers, 1],marker='x', color='blue',label="Sample")
        plt.scatter(x=self.data[self.outliers, 0], y=self.data[self.outliers, 1],
                    color='red', marker='^', label="Outliers")
        plt.scatter(x=self.centroids[0][0], y=self.centroids[0][1],
                    color='black', label="Centroid")
        plt.legend()
        plt.tight_layout()
        plt.show()

class NearestNeighborContextModel:

    def __init__(self,data,n_neighbors=5):
        self.data = data
        self.knn = NearestNeighbors(n_neighbors=n_neighbors,algorithm="ball_tree")
        self.knn.fit(data)
        self.distances, self.indices = self.knn.kneighbors(self.data)
        self.distances = self.distances.mean(axis=1)

        #self.threshold_upper = self.detect_outlier_threshold()
        self.threshold_upper = self.detect_outlier_threshold()

        self.outliers = np.where((self.distances > self.threshold_upper))

    def plot(self):
        data = self.data
        plt.scatter(x=data[:,0], y=data[:,1],color='b',marker='x')
        plt.scatter(x=data[self.outliers,0], y=data[self.outliers,1], color='red', marker='^')
        plt.show()

        bound_upper_mean = self.threshold_upper
        x = np.arange(len(self.distances)).reshape(-1, 1)
        y = self.distances
        plt.plot(y, color='blue')
        plt.plot(x, [bound_upper_mean] * len(x), label='THRESHOLD', linestyle='-',
                 color='red')  # Horizontal line for mean of upper bound
        plt.tight_layout()
        plt.show()

    def moving_average(self, data, window_size):
        """ Returns a simple moving average of the data. """
        return pd.Series(data).rolling(window=window_size, min_periods=1, center=True).mean().values

    def detect_outlier_threshold_knee(self):
        sorted_distances = np.sort(self.distances)
        knee = KneeLocator(range(len(sorted_distances)), sorted_distances, curve='concave', direction='increasing')

        # Get the knee point (index) and corresponding distance threshold
        knee_index = knee.knee
        knee_distance = sorted_distances[knee_index] if knee_index is not None else None
        print(knee_distance)
        return knee_distance
    def detect_outlier_threshold_pecentile(self):
        print(self.distances)
        sorted_distances = self.distances
        threshold = np.percentile(sorted_distances, 90)
        # Plot the sorted distances with the 90th, 95th, and 99th percentile thresholds
        plt.figure(figsize=(8, 5))
        plt.plot(sorted_distances, color='blue', label='Sorted Distances')
        plt.axhline(y=threshold, color='red', linestyle='--', label='95th Percentile')
        plt.xlabel("Data Points (Sorted by Distance)")
        plt.ylabel("Distance")
        plt.legend()
        plt.title("Sorted Distances with 90th, 95th, and 99th Percentile Thresholds")
        plt.show()

        return threshold

    def detect_outlier_threshold(self):
        qr_upper = QuantileRegressor(quantile=0.95)
        qr_lower = QuantileRegressor(quantile=0.05)

        # reshape distances
        x = np.arange(len(self.distances)).reshape(-1, 1)
        #y = self.moving_average(self.distances,int(len(self.distances)* 0.05))
        y = self.distances

        qr_upper.fit(x,y)
        qr_lower.fit(x,y)
        bound_upper = qr_upper.predict(x)
        bound_lower = qr_lower.predict(x)

        return np.mean(bound_upper,axis=0)



class Profile:
    def __init__(self,context,name,method="quantile",titles=[]):
        self.method = method
        self.context = context
        self.name = name
        self.titles = titles
        self.data = {}
        self.contextMeasure = {}
        self.outliers = {}
        self.model = None
    def set_data(self,type,sample):
        self.data[type] = sample

    def compute_context(self):
        # compute the staticial measure for context
        type_max_min = {
            "tempo": {
                "min": 178,
                "max": 24,
            },
            "loudness":{
                "min": 55,
                "max": 0,
            }
        }
        for type in self.data.keys():
            # case for quantile regression 1D
            if self.method == "quantile":
                qr = RegressionContextModel(self.data[type],norm_max=type_max_min[type]["min"],norm_min=type_max_min[type]["max"])
                self.contextMeasure[type] = {}
                self.outliers[type] = {}
                self.contextMeasure[type]["bound_low"] = qr.lower_bound
                self.contextMeasure[type]["bound_high"] = qr.upper_bound
                self.contextMeasure[type]["mean"] = qr.mean
                self.contextMeasure[type]["bound_range"] = qr.bound_range

                # assign outliers
                self.outliers[type]["boolean"] = qr.get_outliers()
            # case for gmm clustering model (multi-dimentional)
            if self.method == "gmm":
                self.model = GaussianContextModel(self.data[type])
                self.contextMeasure[type] = {}
                self.contextMeasure[type]["centroids"] = self.model.centroids
                self.contextMeasure[type]["covariances"] = self.model.covariances
                self.contextMeasure[type]["threshold"] = self.model.threshold
                self.contextMeasure[type]["scores"] = self.model.scores

                # assign outliers
                self.outliers[type] = {}
                self.outliers[type]["boolean"] = self.model.get_outliers()
                self.contextMeasure[type]["size"] = gaussian_cluster_area(self.model.centroids[0], self.model.covariances[0], percentile=0.95)

            if self.method == "knn":
                self.model = NearestNeighborContextModel(self.data[type])

    def plot(self):
        self.model.plot()
    def get_outlier_index(self,type):
        return self.outliers[type]
    def get_outlier(self,type):
        outlier = []
        for idx,val in enumerate(self.data[type]):
            if self.outliers[type]["boolean"][idx]:
                outlier.append(val)
        return np.array(outlier)
    def get_outliers_titles(self,type):
        outlier = []
        for idx, val in enumerate(self.data[type]):
            if self.outliers[type]["boolean"][idx]:
                outlier.append(self.titles[idx])
        return np.array(outlier)

class Outlier:
    def __init__(self,context,value,titles=[]):
        self.context = context
        self.value = value
        self.titles = titles


def get_artist_feature(artist, x="tempo", normalize=False, norm_min=0, norm_max=1):
    # Load the dataset
    songs = msd.loadMSDCsvData("/Users/nurupo/Desktop/dev/msd/song_v2.csv")
    df = pd.DataFrame(songs)

    # Filter data for the specific artist
    df_artist = df[df['artist'] == artist]
    X = df_artist[x].values  # Extract the feature values as a NumPy array

    if normalize:
        # Normalize based on custom domain-specific range [norm_min, norm_max]
        if norm_max != norm_min:  # Avoid division by zero
            X = (X - norm_min) / (norm_max - norm_min)
        else:
            # If norm_max == norm_min, return an array of zeros
            X = np.zeros_like(X)

    return np.array(X)

def fit_outlier_context_quantile(feature_type,outlier_val,context:Profile):
    mean = context.contextMeasure[feature_type]["mean"]
    up = np.mean(context.contextMeasure[feature_type]["bound_high"])
    low = np.mean(context.contextMeasure[feature_type]["bound_low"])

    if outlier_val >= low and outlier_val <= up:
        dist = np.abs(outlier_val-mean)
        return dist
    return False
# 2-dimentional or higher

def fit_outlier_context_cluster(feature_type, outlier_val, context, debug=False):
    # Get contextual properties
    centroids = context.contextMeasure[feature_type]["centroids"]
    covs = context.contextMeasure[feature_type]["covariances"]
    threshold = context.contextMeasure[feature_type]["threshold"]

    # Calculate the Mahalanobis distance threshold based on the chi-square distribution for 2D
    chi_square_threshold = chi2.ppf(1 - threshold, df=2)
    distance_threshold = np.sqrt(chi_square_threshold)

    found_cluster = False
    for i, (centroid, cov) in enumerate(zip(centroids, covs)):
        # Calculate inverse covariance matrix for Mahalanobis distance
        inv_cov = np.linalg.inv(cov)

        # Calculate Mahalanobis distance between the outlier point and the current centroid
        distance = mahalanobis(outlier_val, centroid, inv_cov)

        # Check if the Mahalanobis distance is within the threshold
        if distance <= distance_threshold:
            found_cluster = True
            if debug:
                # Plot only if debug=True
                plt.figure(figsize=(8, 6))
                ax = plt.gca()
                plt.scatter(context.data[feature_type][:, 0], context.data[feature_type][:, 1], color='b', marker='x',
                            label='Sample')

                # Plot centroid
                plt.plot(centroid[0], centroid[1], 'bo', label=f'Centroid {i + 1}')

                # Plot ellipse representing the boundary for this cluster
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                axis_length = 2 * np.sqrt(eigenvalues) * distance_threshold
                angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
                ellipse = plt.matplotlib.patches.Ellipse(
                    centroid, axis_length[0], axis_length[1], angle=angle,
                    edgecolor='green', fc='none', lw=2, label='Cluster Boundary'
                )
                ax.add_patch(ellipse)

                # Plot the outlier point
                plt.plot(outlier_val[0], outlier_val[1], 'ro', markersize=10, label='Outlier (Inside Cluster)')
                plt.text(outlier_val[0], outlier_val[1], f'Distance={distance:.2f}', color='red')

                # Plot title and labels
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
                plt.title(f'Outlier Context Check for {feature_type}')
                plt.legend()
                plt.tight_layout()
                plt.show()
            return distance  # Point is in this cluster

    # If point is not inside any cluster, optionally plot as outside (if debug=True)
    if not found_cluster and debug:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        plt.scatter(context.data[feature_type][:, 0], context.data[feature_type][:, 1], color='b', marker='x',
                    label='Sample')

        # Plot centroids and boundaries
        for i, (centroid, cov) in enumerate(zip(centroids, covs)):
            plt.plot(centroid[0], centroid[1], 'bo', label=f'Centroid {i + 1}' if i == 0 else "")
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            axis_length = 2 * np.sqrt(eigenvalues) * distance_threshold
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            ellipse = plt.matplotlib.patches.Ellipse(
                centroid, axis_length[0], axis_length[1], angle=angle,
                edgecolor='green', fc='none', lw=2, label='Cluster Boundary' if i == 0 else ""
            )
            ax.add_patch(ellipse)

        # Plot the outlier point outside any cluster
        plt.plot(outlier_val[0], outlier_val[1], 'rx', markersize=10, label='Outlier (Outside Clusters)')
        plt.text(outlier_val[0], outlier_val[1], 'Outside', color='red')

        # Plot title and labels
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Outlier Context Check for {feature_type}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return False  # Point is not in any cluster


def gaussian_cluster_area(centroid, cov, percentile=0.95):
    """
    Calculate the area of the Gaussian cluster's context boundary.

    Parameters:
        centroid (array): The mean (centroid) of the cluster.
        cov (array): The covariance matrix of the cluster.
        percentile (float): The desired percentile for the boundary (e.g., 0.95).

    Returns:
        float: The area of the cluster boundary in 2D.
    """
    # Dimensionality of the data
    d = len(centroid)

    # Calculate the Mahalanobis distance threshold based on the chi-square distribution
    chi_square_threshold = chi2.ppf(percentile, df=d)

    # Calculate the volume of the ellipsoid for d-dimensional space
    # For 2D, the "volume" is the area of the ellipse
    eigenvalues = np.linalg.eigvalsh(cov)  # Get eigenvalues of covariance matrix
    area = np.pi * chi_square_threshold * np.sqrt(np.prod(eigenvalues))
    return area


# 1-dim
def experiment_quantile_regreesion():
    songs = msd.loadMSDCsvData("/Users/nurupo/Desktop/dev/msd/song_v2.csv")
    df = pd.DataFrame(songs)
    all_artists = df['artist'].unique().tolist()
    print(len(all_artists))
    feature_type = "tempo"
    # bound_sparsity_therehold = 0.6 #Tempo
    bound_sparsity_therehold = 25
    csv_output = {
        "outlier_value": [],
        "outlier_title": [],
        "origin_context": [],
        "low_bound": [],
        "high_bound": [],
        "mean": [],
        "fit_context": [],
        "new_low_bound": [],
        "new_high_bound": [],
        "new_mean": [],
        "score": [],
    }
    # 1. get outliers for each context
    identify_outlier = []
    all_contexts = []
    for artist in all_artists:
        x = get_artist_feature(artist, x=feature_type, normalize=False)
        titles = get_artist_feature(artist, x="title")
        p = Profile(context=artist, name=artist,titles=titles)
        p.set_data(feature_type, x)
        p.compute_context()

        outliers = p.get_outlier(feature_type)
        bound_sparsity = p.contextMeasure[feature_type]["bound_range"]
        if len(outliers) > 0 and bound_sparsity < bound_sparsity_therehold:
            # print(np.mean(p.contextMeasure[feature_type]["bound_low"]))
            # print(np.mean(p.contextMeasure[feature_type]["bound_high"]))
            # print(p.contextMeasure[feature_type]["mean"])
            # print(artist,p.contextMeasure[feature_type]["bound_range"])
            # print(outliers)
            print(f"processed: {artist}, {len(outliers)} outlier found.")
            outliers_title = p.get_outliers_titles(feature_type)
            identify_outlier.append(Outlier(context=p, value=outliers, titles=outliers_title))
            all_contexts.append(p)
        else:
            print(f"No outlier found for / bounds skipped for {artist}")

    # 2. find relationship cross context
    # Find relationship across contexts
    for outlier in identify_outlier:
        for idx,val in enumerate(outlier.value):
            context_fit_score = []  # List to store fit scores

            for context in all_contexts:
                if outlier.context == context:
                    continue

                # Calculate the fit (distance) for the current context
                fit = fit_outlier_context_quantile(feature_type, val, context)
                if fit:
                    context_fit_score.append((fit, context))  # Append tuple of fit and context name

            # Convert fit scores to numpy array for easy indexing
            fit_scores = np.array([fit[0] for fit in context_fit_score])
            context_list = [fit[1] for fit in context_fit_score]

            # Find the index of the minimum fit score
            if fit_scores.size > 0:
                min_index = np.argmin(fit_scores)
                best_fit_context = context_list[min_index]
                lowest_fit = fit_scores[min_index]

                # Print and store results
                # print(f"Outlier value {val}: Context fit scores -> {context_fit_score}")
                title = outlier.titles[idx]
                print(f"Found: {val}({title}) from {outlier.context.name} -> {best_fit_context.name} with fit score {lowest_fit}")
                # append to csv
                csv_output["outlier_value"].append(val)
                csv_output["outlier_title"].append(title)
                csv_output["origin_context"].append(outlier.context.name)
                csv_output["low_bound"].append(np.mean(outlier.context.contextMeasure[feature_type]["bound_low"]))
                csv_output["high_bound"].append(np.mean(outlier.context.contextMeasure[feature_type]["bound_high"]))
                csv_output["mean"].append(outlier.context.contextMeasure[feature_type]["mean"])
                csv_output["fit_context"].append(best_fit_context.name)
                csv_output["new_low_bound"].append(np.mean(best_fit_context.contextMeasure[feature_type]["bound_low"]))
                csv_output["new_high_bound"].append(
                    np.mean(best_fit_context.contextMeasure[feature_type]["bound_high"]))
                csv_output["new_mean"].append(best_fit_context.contextMeasure[feature_type]["mean"])
                csv_output["score"].append(lowest_fit)

    df = pd.DataFrame.from_dict(csv_output)
    df.to_csv("./in-album_outlier_tempo_2.csv")

    # # finally print detected outlier
    # print("found outliers")
    # for outlier in identify_outlier:
    #     print(outlier.context)
    #     print(outlier.value)

    # Outlier Value | Origin Context | High Bound | Low Bound | New Context | High Bound | Low Bound | Fit Score

# muti-dimentional
def experiment_clustering():
    ## PARAMETER TO SET
    feature_type = "loudness_tempo"
    context_size_threshold = 2000
    identify_outlier = []
    all_contexts = []


    # INIT DATASET
    songs = msd.loadMSDCsvData("/Users/nurupo/Desktop/dev/msd/song_v2.csv")
    df = pd.DataFrame(songs)
    all_artists = df['artist'].unique().tolist()
    #all_artists = ["Chezidek","MNEMIC","Blue Oyster Cult"]
    # initial csv for outputing
    csv_output = {
        "outlier_value": [],
        "outlier_title": [],
        "origin_context": [],
        "centroid": [],
        "covariance": [],
        "fit_context": [],
        "new_centroid": [],
        "new_covariance": [],
        "score": [],
    }

    # 1. get outliers for each context
    for artist in all_artists:
        print(f"Processing artist {artist}")
        x = get_artist_feature(artist, x="tempo", normalize=False)
        y = get_artist_feature(artist, x="loudness", normalize=False)
        titles = get_artist_feature(artist, x="title")
        if len(x) < 2:
            print(f"Skipping artist {artist} with lower 2 samples!")
            continue
        p = Profile("artist",artist,method="gmm",titles=titles)
        # pack feature
        data = np.array(list(zip(x, y)))
        p.set_data(feature_type,data)
        p.compute_context()
        outliers = p.get_outlier(feature_type)


        outliers_title = p.get_outliers_titles(feature_type)
        identify_outlier.append(Outlier(context=p, value=outliers,titles=outliers_title))
        all_contexts.append(p)
        #p.plot()

    # 2. find relationship cross context
    # Find relationship across contexts
    for outlier in identify_outlier:
        for idx,val in enumerate(outlier.value):
            context_fit_score = []  # List to store fit scores

            for context in all_contexts:
                # ignore self
                if outlier.context == context:
                    continue
                # ignore if context size is too large (which means it covers lots of area)
                # Todo -> Done! Check & filter contextual sparsity ...
                if context.contextMeasure[feature_type]["size"] > context_size_threshold:
                    print(f"{context.name} skipped, size too sparse: {context.contextMeasure[feature_type]['size']}")
                    continue
                # Calculate the fit (distance) for the current context
                fit = fit_outlier_context_cluster(feature_type, val, context,debug=False)
                if fit:
                    context_fit_score.append((fit, context))  # Append tuple of fit and context name

            # Convert fit scores to numpy array for easy indexing
            fit_scores = np.array([fit[0] for fit in context_fit_score])
            context_list = [fit[1] for fit in context_fit_score]

            # Find the index of the minimum fit score
            if fit_scores.size > 0:
                min_index = np.argmin(fit_scores)
                best_fit_context = context_list[min_index]
                lowest_fit = fit_scores[min_index]
                title = outlier.titles[idx]
                print(f"Found: {val}({title}) from {outlier.context.name} -> {best_fit_context.name} with fit score {lowest_fit}")
                """
                    "origin_context": [],
                    "centroid": [],
                    "covariance": [],
                    "fit_context": [],
                    "new_centroid": [],
                    "new_covariance": [],
                    "score": [],
                """

                csv_output["outlier_value"].append(val)
                csv_output["outlier_title"].append(title)
                csv_output["origin_context"].append(outlier.context.name)
                csv_output["centroid"].append(outlier.context.contextMeasure[feature_type]["centroids"][0])
                csv_output["covariance"].append(outlier.context.contextMeasure[feature_type]["covariances"][0])
                csv_output["fit_context"].append(best_fit_context.name)
                csv_output["new_centroid"].append(best_fit_context.contextMeasure[feature_type]["centroids"][0])
                csv_output["new_covariance"].append(best_fit_context.contextMeasure[feature_type]["covariances"][0])
                csv_output["score"].append(lowest_fit)
    df = pd.DataFrame.from_dict(csv_output)
    df.to_csv("./cross-artist_outlier_gaussian_loudness_tempo.csv")
    print(df)

def plotArtistDebug(artist):
    # debug
    feature_type = "loudness_tempo"
    x = get_artist_feature(artist, x="tempo", normalize=False)
    y = get_artist_feature(artist, x="loudness", normalize=False)
    titles = get_artist_feature(artist, x="title")

    p = Profile("artist", artist, method="gmm", titles=titles)
    # pack feature
    data = np.array(list(zip(x, y)))
    p.set_data(feature_type, data)
    p.compute_context()
    print(p.contextMeasure[feature_type]["size"])
    p.plot()




if __name__ == '__main__':
    experiment_quantile_regreesion()
    #experiment_clustering()

    #plotArtistDebug("Meic Stevens")
    #plotArtistDebug("Blue Oyster Cult")
    # songs = msd.loadMSDCsvData("/Users/nurupo/Desktop/dev/msd/song_v2.csv")
    # df = pd.DataFrame(songs)
    # all_style = df['genre'].unique().tolist()
    # print(all_style,len(all_style))
