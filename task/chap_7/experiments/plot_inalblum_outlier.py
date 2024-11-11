import re
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

import feature.msd_dataset as msd
import numpy as np

def extract_numbers(s):
    return re.findall(r'-?\d+\.?\d*', s)
def plot_outliers(target_genre="dance and electronica"):
    # Load outlier data from two separate CSV files
    csv_tempo_outlier = pd.read_csv("/Users/nurupo/Desktop/dev/Project-Outlier-Music/task/chap_7/in-album_outlier_tempo.csv")
    csv_loudness_outlier = pd.read_csv("/Users/nurupo/Desktop/dev/Project-Outlier-Music/task/chap_7/in-album_outlier_loudness.csv")

    songs = msd.loadMSDCsvData("/Users/nurupo/Desktop/dev/msd/song_v2.csv")
    df = pd.DataFrame(songs)

    # Filter songs DataFrame for the specific genre
    df_genre = df[df['genre'] == target_genre]

    # Initialize lists to store data
    outlier_data = []
    reference_data = []

    # Merge outlier dataframes on a common key, adding appropriate suffixes
    merged_outliers = pd.merge(csv_tempo_outlier, csv_loudness_outlier, on="origin_context", suffixes=('_tempo', '_loudness'))

    for index, row in merged_outliers.iterrows():
        artist = row["origin_context"]
        # Filter data for the specific artist
        df_artist = df_genre[df_genre['artist'] == artist]
        if not df_artist.empty:
            x = float(row["outlier_value_tempo"])  # Tempo values
            y = float(row["outlier_value_loudness"])  # Loudness values
            outlier_data.append([x, y])

    # Extract reference data from filtered genre data
    for index, row in df_genre.iterrows():
        x, y = row["tempo"], row["loudness"]
        reference_data.append([x, y])

    outlier_data = np.array(outlier_data)
    reference_data = np.array(reference_data)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))  # Create a figure with 1 row and 2 columns

    # KDE for Tempo values
    if len(reference_data) > 0:
        sns.kdeplot(reference_data[:, 0], ax=ax[0], color="blue", fill=True, label='Dataset Tempo')
    if len(outlier_data) > 0:
        sns.kdeplot(outlier_data[:, 0], ax=ax[0], color="red", fill=True, label='Outlier Tempo')
    ax[0].set_title('KDE of Tempo')
    ax[0].set_xlabel('Tempo')
    ax[0].set_ylabel('Density')
    ax[0].legend()

    # KDE for Loudness values
    if len(reference_data) > 0:
        sns.kdeplot(reference_data[:, 1], ax=ax[1], color="blue", fill=True, label='Dataset Loudness')
    if len(outlier_data) > 0:
        sns.kdeplot(outlier_data[:, 1], ax=ax[1], color="red", fill=True, label='Outlier Loudness')
    ax[1].set_title('KDE of Loudness')
    ax[1].set_xlabel('Loudness')
    ax[1].set_ylabel('Density')
    ax[1].legend()
    plt.suptitle(target_genre, fontsize=16,fontweight='bold')
    plt.tight_layout()  # Adjust layout to not overlap
    plt.show()


def plot_all_genres_outliers(genres):
    csv_tempo_outlier = pd.read_csv(
        "/Users/nurupo/Desktop/dev/Project-Outlier-Music/task/chap_7/in-album_outlier_tempo.csv")
    csv_loudness_outlier = pd.read_csv(
        "/Users/nurupo/Desktop/dev/Project-Outlier-Music/task/chap_7/in-album_outlier_loudness.csv")

    songs = msd.loadMSDCsvData("/Users/nurupo/Desktop/dev/msd/song_v2.csv")
    df = pd.DataFrame(songs)

    # Merge outlier dataframes on a common key
    merged_outliers = pd.merge(csv_tempo_outlier, csv_loudness_outlier, on="origin_context",
                               suffixes=('_tempo', '_loudness'))

    # Setup subplot grid - Fixed to handle all genres
    n_genres = len(genres)
    n_cols = 2  # Two columns: tempo and loudness
    n_rows = n_genres  # One row per genre - this is the key fix

    # Create figure with proper size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2 * n_rows), squeeze=False)

    for i, genre in enumerate(genres):
        # Get the correct row for the current genre
        ax_tempo = axes[i, 0]  # Changed from i//2 to i
        ax_loudness = axes[i, 1]  # Changed from i//2 to i

        # Clear previous plots
        ax_tempo.clear()
        ax_loudness.clear()

        # Get genre-specific data
        df_genre = df[df['genre'] == genre].copy()
        relevant_artists = df_genre['artist'].unique()

        # Get separate outliers for tempo and loudness
        filtered_tempo_outliers = csv_tempo_outlier[csv_tempo_outlier['origin_context'].isin(relevant_artists)].copy()
        filtered_loudness_outliers = csv_loudness_outlier[
            csv_loudness_outlier['origin_context'].isin(relevant_artists)].copy()

        # Separate tempo and loudness data
        # Normal distributions
        normal_tempo = df_genre['tempo'].values
        normal_loudness = df_genre['loudness'].values

        # Outlier distributions
        outlier_tempo = filtered_tempo_outliers['outlier_value'].astype(float).values
        outlier_loudness = filtered_loudness_outliers['outlier_value'].astype(float).values

        # Plot Tempo distributions (left subplot)
        if len(normal_tempo) > 0:
            sns.kdeplot(data=normal_tempo, ax=ax_tempo, color="blue",
                        fill=True, label=f'Normal Tempo')
        if len(outlier_tempo) > 0:
            sns.kdeplot(data=outlier_tempo, ax=ax_tempo, color="red",
                        fill=True, label=f'Outlier Tempo')

        ax_tempo.set_title(f'Tempo Distribution - {genre}')
        ax_tempo.set_xlabel('Tempo (BPM)')
        ax_tempo.set_ylabel('Density')
        ax_tempo.legend()

        # Plot Loudness distributions (right subplot)
        if len(normal_loudness) > 0:
            sns.kdeplot(data=normal_loudness, ax=ax_loudness, color="blue",
                        fill=True, label=f'Normal Loudness')
        if len(outlier_loudness) > 0:
            sns.kdeplot(data=outlier_loudness, ax=ax_loudness, color="red",
                        fill=True, label=f'Outlier Loudness')
        ax_loudness.set_title(f'Loudness Distribution - {genre}')
        ax_loudness.set_xlabel('Loudness (dB)')
        ax_loudness.set_ylabel('Density')
        ax_loudness.legend()

        # Print detailed statistics for verification
        print(f"\nGenre: {genre}")
        print("Tempo Statistics:")
        print(f"  Normal songs: {len(normal_tempo)}")
        print(f"  Tempo outliers: {len(outlier_tempo)}")
        print(f"  Tempo outlier ratio: {len(outlier_tempo) / len(normal_tempo):.2%}")

        print("Loudness Statistics:")
        print(f"  Normal songs: {len(normal_loudness)}")
        print(f"  Loudness outliers: {len(outlier_loudness)}")
        print(f"  Loudness outlier ratio: {len(outlier_loudness) / len(normal_loudness):.2%}")

        # Cleanup for next iteration
        del df_genre
        del filtered_tempo_outliers
        del filtered_loudness_outliers
        del normal_tempo, normal_loudness
        del outlier_tempo, outlier_loudness

    plt.tight_layout()
    plt.show()
def plot_outliers_use_algorthm_mean():
    # Load outlier data from two separate CSV files
    csv_tempo_outlier = pd.read_csv(
        "/Users/nurupo/Desktop/dev/Project-Outlier-Music/task/chap_7/in-album_outlier_tempo.csv")
    csv_loudness_outlier = pd.read_csv(
        "/Users/nurupo/Desktop/dev/Project-Outlier-Music/task/chap_7/in-album_outlier_loudness.csv")

    # Load the full songs dataset
    songs = msd.loadMSDCsvData("/Users/nurupo/Desktop/dev/msd/song_v2.csv")
    df_songs = pd.DataFrame(songs)

    # Filter for the target genre
    target_genre = "punk"
    relevant_artists = df_songs[df_songs['genre'] == target_genre]['artist'].unique()

    # Filter the outlier dataframes based on these artists
    csv_tempo_outlier = csv_tempo_outlier[csv_tempo_outlier['origin_context'].isin(relevant_artists)]
    csv_loudness_outlier = csv_loudness_outlier[csv_loudness_outlier['origin_context'].isin(relevant_artists)]

    # Initialize lists to store data
    outlier_data = []
    norm_tempo_data = []
    norm_loudness_data = []

    # Merge outlier dataframes on a common key, adding appropriate suffixes
    merged_outliers = pd.merge(csv_tempo_outlier, csv_loudness_outlier, on="origin_context",
                               suffixes=('_tempo', '_loudness'))
    for index, row in merged_outliers.iterrows():
        x = float(row["outlier_value_tempo"])  # Tempo values
        y = float(row["outlier_value_loudness"])  # Loudness values
        outlier_data.append([x, y])

        # Append the corresponding mean values for each outlier
        norm_tempo_data.append(float(row["mean_tempo"]))
        norm_loudness_data.append(float(row["mean_loudness"]))

    # Convert to numpy array for easier plotting
    outlier_data = np.array(outlier_data)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Create a figure with 1 row and 2 columns

    # KDE for Tempo values
    sns.kdeplot(outlier_data[:, 0], ax=ax[0], color="red", fill=True, label='Outlier Tempo')
    sns.kdeplot(norm_tempo_data, ax=ax[0], color="blue", fill=True, label='Norm Tempo')  # Norm distribution
    ax[0].set_title('KDE of Tempo')
    ax[0].set_xlabel('Tempo')
    ax[0].set_ylabel('Density')
    ax[0].legend()

    # KDE for Loudness values
    sns.kdeplot(outlier_data[:, 1], ax=ax[1], color="red", fill=True, label='Outlier Loudness')
    sns.kdeplot(norm_loudness_data, ax=ax[1], color="blue", fill=True, label='Norm Loudness')  # Norm distribution
    ax[1].set_title('KDE of Loudness')
    ax[1].set_xlabel('Loudness')
    ax[1].set_ylabel('Density')
    ax[1].legend()

    plt.tight_layout()  # Adjust layout to not overlap
    plt.show()

if __name__ == '__main__':
    songs = msd.loadMSDCsvData("/Users/nurupo/Desktop/dev/msd/song_v2.csv")
    df = pd.DataFrame(songs)
    all_style = df['genre'].unique().tolist()

    all_style = [
        "pop",
        "hip-hop",
        "dance and electronica",
        "metal",
        "classical",
    ]
    print(all_style)
    # for genre in all_style:
    #     print(genre)
    #     plot_outliers(target_genre=genre)

    plot_all_genres_outliers(all_style)
    #plot_outliers_use_algorthm_mean()