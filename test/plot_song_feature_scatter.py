import feature.analysis
from feature.analysis import *
from model.song import *
from plot.chord_transition_plot import *
from pychord import Chord
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN

if __name__ == '__main__':
    TARGET_MODE = "major"

    PATH = "/Users/nurupo/Desktop/dev/music4all/"
    audio_folder = "audios"
    path_csv = os.path.join(PATH, "stratified_songs_pop.csv")

    song_collections = []
    song_outliers = []
    song_feature_x = []
    song_feature_y = []
    song_feature_z = []
    song_ids = []
    CURRENT_ID = 0
    ARIST_ID_MAP = {}
    if os.path.exists(path_csv):
        csv = pd.read_csv(path_csv)
        for _, item in csv.iterrows():
            id = item['id']
            popularity = item['popularity']
            valence = item['valence']
            energy = item['energy']
            danceability = item['danceability']
            h5_path = os.path.join(PATH, audio_folder, id + ".h5")
            song = Song.from_h5(h5_path)
            artist = song.artist
            n_tonic_chords = 0
            n_chords = 0
            n_chords_pattern = 0
            n_non_diatonic = 0

            # chords_array = [c[2] for c in song.chord_transposed if c[2] != "N"]
            # _,_, n_non_diatonic = analysis.anlysisromanMumerals(chords_array,song.mode == "major")
            #
            for c in song.chord_transposed:
                time, beat, chord = c
                chord = chord.replace(":", "")
                if chord == "N": continue  # skip empty chord
                n_chords += 1
                if song.mode == "major":
                    if chord == "Cmaj": n_tonic_chords += 1

                else:
                    if chord == "Amin": n_tonic_chords += 1
            for ptn in song.chord_pattern:
                n_chords_pattern += len(ptn["pattern"]) * ptn["matches"]
                n_non_diatonic += ptn["non_diatonic_chords_count"]
            if n_chords_pattern == 0:continue
            if n_chords == 0:continue
            # Embed ID
            if song.artist not in ARIST_ID_MAP:
                ARIST_ID_MAP[song.artist] = CURRENT_ID
                CURRENT_ID+= 1

            song_collections.append(song)
            song_feature_x.append(n_non_diatonic)
            song_feature_y.append(song.release)
            song_feature_z.append(song.tempo)
            song_ids.append(id)


    # Convert song features to a DataFrame for easier manipulation
    df_features = pd.DataFrame({
        'id': song_ids,
        'x': song_feature_x,
        'y': song_feature_y
    })

    # Convert song features to a NumPy array for DBSCAN
    features = np.array(list(zip(song_feature_x, song_feature_y)))

    # Apply DBSCAN to the dataset
    db = DBSCAN(eps=3, min_samples=10).fit(features)
    labels = db.labels_

    # Identify points that are classified as noise by DBSCAN
    outlier_mask = labels == -1

    # Update the outliers list
    song_outliers = [song_collections[i] for i in range(len(song_collections)) if outlier_mask[i]]

    # Convert song features to a DataFrame for easier manipulation
    df_features = pd.DataFrame({
        'id': song_ids,
        'x': song_feature_x,
        'y': song_feature_y,
        'Outlier': outlier_mask
    })

    df_outliers = pd.DataFrame({
        'id': [song.id for song in song_outliers],
        'title': [song.title for song in song_outliers],
        'artist': [song.artist for song in song_outliers],
        'release': [song.release for song in song_outliers],
        'key': [song.key for song in song_outliers],
        'mode': [song.mode for song in song_outliers],
        'patterns': [song.chord_pattern for song in song_outliers],
        'n_nondiatonic': [sum(map(lambda ptn: ptn["non_diatonic_chords_count"], song.chord_pattern)) for song in
                          song_outliers],
    })

    df_outliers.to_csv("/Users/nurupo/Desktop/dev/music4all/non_diatonic.csv", index=False)

    print(df_outliers)
    plt.figure(figsize=(6, 5))
    gs = GridSpec(2, 1, height_ratios=[3, 1])

    # Scatter Plot on top
    ax_scatter = plt.subplot(gs[0])
    # Color points based on the 'Outlier' column
    scatter_colors = df_features['Outlier'].map({True: 'red', False: 'blue'})
    ax_scatter.scatter(df_features['x'], df_features['y'], c=scatter_colors, alpha=0.6)
    ax_scatter.set_xlabel('Number of Non-Diatonic Chords')
    ax_scatter.set_ylabel('Release Year')
    ax_scatter.set_title('Scatter Plot of Non-Diatonic Chords by Release Year')
    ax_scatter.grid(True)

    # KDE Plot at the bottom
    ax_kde = plt.subplot(gs[1])
    # Plot KDE for all points
    sns.kdeplot(data=df_features, x='x',y='y', fill=True, bw_adjust=0.5, ax=ax_kde, color='blue')
    ax_kde.set_xlabel('Number of Non-Diatonic Chords')
    ax_kde.set_ylabel('Density')
    ax_kde.set_title('Kernel Density Estimate Plot of Non-Diatonic Chords')

    plt.tight_layout()
    plt.show()

    print("DBSCAN Outlier Detection Complete")