import os
import pandas as pd
from model.song import Song
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration for year intervals
YEAR_INTERVAL = 5

def process_song(song, non_diatonic_counts_by_year, year_interval):
    year = int(song.release)
    # Group the year into intervals
    year_group = (year // year_interval) * year_interval
    for pattern in song.chord_pattern:
        if "N" in pattern.get("pattern"):
            print("SKIP INVALID CHORD DATA")
            continue
        non_diatonic_count = pattern.get("non_diatonic_chords_count", 0)
        if non_diatonic_count > 0:
            non_diatonic_counts_by_year[year_group] = non_diatonic_counts_by_year.get(year_group, 0) + 1

if __name__ == '__main__':
    PATH = "/Users/nurupo/Desktop/dev/music4all/"
    audio_folder = "audios"
    path_csv = os.path.join(PATH, "stratified_songs_pop.csv")

    non_diatonic_counts_by_year_minor = {}
    non_diatonic_counts_by_year_major = {}

    if os.path.exists(path_csv):
        csv = pd.read_csv(path_csv)
        for _, item in csv.iterrows():
            id = item['id']
            h5_path = os.path.join(PATH, audio_folder, id + ".h5")
            song = Song.from_h5(h5_path)

            if song.mode == "minor":
                process_song(song, non_diatonic_counts_by_year_minor, YEAR_INTERVAL)
            elif song.mode == "major":
                process_song(song, non_diatonic_counts_by_year_major, YEAR_INTERVAL)

    # Convert dictionaries to DataFrame and sort by Year
    df_non_diatonic_counts_minor = pd.DataFrame(list(non_diatonic_counts_by_year_minor.items()),
                                                columns=['Year', 'Count_of_Songs_with_Non_Diatonic_Chords']).sort_values(by='Year')
    df_non_diatonic_counts_major = pd.DataFrame(list(non_diatonic_counts_by_year_major.items()),
                                                columns=['Year', 'Count_of_Songs_with_Non_Diatonic_Chords']).sort_values(by='Year')

    # plot
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))

    # Plot for 'minor' mode
    sns.barplot(x='Year', y='Count_of_Songs_with_Non_Diatonic_Chords', data=df_non_diatonic_counts_minor, ax=axes[0], color="tab:red")
    sns.regplot(x=np.arange(len(df_non_diatonic_counts_minor)), y='Count_of_Songs_with_Non_Diatonic_Chords', data=df_non_diatonic_counts_minor, ax=axes[0],
                scatter=False, color="tab:red", order=3, ci=None)
    axes[0].set_title('Count of Songs with Non-Diatonic Chords by Year Interval (Minor)')
    axes[0].set_xticklabels(df_non_diatonic_counts_minor['Year'], rotation=0)
    axes[0].set_xlabel('Year (Interval 5)')
    axes[0].set_ylabel('Non-diatonic Songs')

    # Plot for 'major' mode
    sns.barplot(x='Year', y='Count_of_Songs_with_Non_Diatonic_Chords', data=df_non_diatonic_counts_major, ax=axes[1], color="tab:blue")
    sns.regplot(x=np.arange(len(df_non_diatonic_counts_major)), y='Count_of_Songs_with_Non_Diatonic_Chords', data=df_non_diatonic_counts_major, ax=axes[1],
                scatter=False, color="tab:blue", order=3, ci=None)
    axes[1].set_title('Non-Diatonic Songs vs Year (Major)')
    axes[1].set_xticklabels(df_non_diatonic_counts_major['Year'], rotation=0)
    axes[1].set_xlabel('Year (Interval 5)')
    axes[1].set_ylabel('Non-diatonic Songs')

    plt.tight_layout()
    plt.show()
