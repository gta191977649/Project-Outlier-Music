import gower
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn_quantile import RandomForestQuantileRegressor

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

ContextFeatureList = [
    "artist",
]

FeatureList = [
    "loudness",
    "tempo",
]


def encode_categorical_features(data, categorical_features):
    le = LabelEncoder()
    encoded_data = data.copy()
    for feature in categorical_features:
        encoded_data[feature] = le.fit_transform(encoded_data[feature])
    return encoded_data, le


def compute_anomaly_scores(data, reference_indices, behavioral_features, contextual_features):
    anomaly_scores = np.zeros(data.shape[0])

    for feature in behavioral_features:
        qrf = RandomForestQuantileRegressor(n_estimators=100, min_samples_leaf=10,q=[0.01, 0.05, 0.1, 0.9, 0.95, 0.99])
        feature_anomaly_scores = np.zeros(data.shape[0])

        for i, neighbors in enumerate(reference_indices.values):
            X_train = data.iloc[neighbors][contextual_features]
            y_train = data.iloc[neighbors][feature]

            qrf.fit(X_train, y_train)

            X_test = data.iloc[i:i + 1][contextual_features]
            y_test = data.iloc[i:i + 1][feature].values[0]

            # Predict quantiles
            quantiles = qrf.predict(X_test)
            quantiles = quantiles.flatten()

            if y_test < quantiles[0]:
                score = (quantiles[0] - y_test) / (quantiles[2] - quantiles[0])
            elif y_test > quantiles[5]:
                score = (y_test - quantiles[5]) / (quantiles[5] - quantiles[3])
            else:
                score = 0

            feature_anomaly_scores[i] = min(score, 0.1)  # Clip the score at 0.1

        anomaly_scores += feature_anomaly_scores

    return anomaly_scores


if __name__ == '__main__':
    N_SAMPLE = 100
    csv = pd.read_csv("/Users/nurupo/Desktop/dev/msd/song_picked.csv")
    csv = csv[:N_SAMPLE]

    csv_context = csv[ContextFeatureList]
    csv_feature = csv[FeatureList]

    # Normalize behavioral features
    #scaler = MinMaxScaler()
    #csv[FeatureList] = scaler.fit_transform(csv_feature)

    # Get gower score for Context Feature (using original categorical data)
    distance_matrix = gower.gower_matrix(csv_context)

    # Get knn reference group for Context Feature
    knn = NearestNeighbors(algorithm='brute', metric="precomputed").fit(distance_matrix)
    indices = knn.kneighbors(distance_matrix, return_distance=False)
    indices = pd.DataFrame(indices)
    indices['point_index'] = indices.index

    # Visualize distance matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, cmap="Greens_r")
    plt.title("Distance Matrix Heatmap")
    plt.show()

    # Encode categorical features for anomaly detection
    encoded_csv, label_encoder = encode_categorical_features(csv, ContextFeatureList)

    # Compute anomaly scores
    anomaly_scores = compute_anomaly_scores(encoded_csv, indices, FeatureList, ContextFeatureList)

    # Add anomaly scores to the DataFrame
    csv['anomaly_score'] = anomaly_scores

    # Sort by anomaly score (descending) and display top 10 anomalies
    top_anomalies = csv.sort_values('anomaly_score', ascending=False).head(10)
    print(top_anomalies[ContextFeatureList + FeatureList + ['anomaly_score']])

    # Visualize results
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(csv['loudness'], csv['tempo'], c=csv['anomaly_score'], cmap='viridis')
    plt.colorbar(scatter, label='Anomaly Score')
    plt.xlabel('Loudness')
    plt.ylabel('Tempo')
    plt.title('Behavioral Features Colored by Anomaly Score')
    plt.show()

    # Distribution of anomaly scores
    plt.figure(figsize=(10, 6))
    sns.histplot(csv['anomaly_score'], kde=True)
    plt.xlabel('Anomaly Score')
    plt.title('Distribution of Anomaly Scores')
    plt.show()


    # Function to explain anomalies
    def explain_anomaly(data, index, reference_indices, behavioral_features, contextual_features):
        neighbors = data.iloc[reference_indices.iloc[index]]
        anomaly = data.iloc[index]

        print(f"Anomaly at index {index}:")
        for feature in contextual_features:
            print(f"{feature}: {anomaly[feature]}")
        print(f"Behavioral features: {anomaly[behavioral_features].to_dict()}")
        print("\nNeighbor statistics:")

        for feature in behavioral_features:
            print(f"\n{feature}:")
            print(f"Anomaly value: {anomaly[feature]}")
            print(f"Neighbor mean: {neighbors[feature].mean()}")
            print(f"Neighbor std: {neighbors[feature].std()}")
            print(f"Neighbor min: {neighbors[feature].min()}")
            print(f"Neighbor max: {neighbors[feature].max()}")


    # Explain the top anomaly
    top_anomaly_index = csv['anomaly_score'].idxmax()
    explain_anomaly(csv, top_anomaly_index, indices, FeatureList, ContextFeatureList)