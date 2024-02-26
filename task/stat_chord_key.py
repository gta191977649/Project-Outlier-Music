import os
import pandas as pd
from model.song import Song
import matplotlib.pyplot as plt
import seaborn as sns

def process_song(song, non_diatonic_counts_by_year):
    year = int(song.release)
    for pattern in song.chord_pattern:
        if "N" in pattern.get("pattern"):
            print("SKIP INVALID CHORD DATA")
            continue
        non_diatonic_count = pattern.get("non_diatonic_chords_count", 0)
        if non_diatonic_count > 0:
            non_diatonic_counts_by_year[year] = non_diatonic_counts_by_year.get(year, 0) + non_diatonic_count

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
                process_song(song, non_diatonic_counts_by_year_minor)
            elif song.mode == "major":
                process_song(song, non_diatonic_counts_by_year_major)

    df_non_diatonic_counts_minor = pd.DataFrame(list(non_diatonic_counts_by_year_minor.items()),
                                                columns=['Year', 'Count_of_Songs_with_Non_Diatonic_Chords'])
    df_non_diatonic_counts_major = pd.DataFrame(list(non_diatonic_counts_by_year_major.items()),
                                                columns=['Year', 'Count_of_Songs_with_Non_Diatonic_Chords'])

    # plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))

    # Plot for 'minor' mode
    axes[0].bar(df_non_diatonic_counts_minor['Year'],
                df_non_diatonic_counts_minor['Count_of_Songs_with_Non_Diatonic_Chords'],
                width=0.4, color="tab:red", zorder=0)
    sns.regplot(x='Year', y='Count_of_Songs_with_Non_Diatonic_Chords', data=df_non_diatonic_counts_minor, ax=axes[0],
                scatter=False, color="tab:red", order=10)
    axes[0].set_title('Count of Songs with Non-Diatonic Chords by Year (Minor)')
    axes[0].set_ylim(0, None)
    axes[0].set_xlim(df_non_diatonic_counts_minor['Year'].min(), df_non_diatonic_counts_minor['Year'].max())
    axes[0].set_xticks(df_non_diatonic_counts_minor['Year'])
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_ylabel('N of Non-diatonic Chords')

    # Plot for 'major' mode
    axes[1].bar(df_non_diatonic_counts_major['Year'],
                df_non_diatonic_counts_major['Count_of_Songs_with_Non_Diatonic_Chords'],
                width=0.4, color="tab:blue", zorder=0)
    sns.regplot(x='Year', y='Count_of_Songs_with_Non_Diatonic_Chords', data=df_non_diatonic_counts_major, ax=axes[1],
                scatter=False, color="tab:blue", order=10)
    axes[1].set_title('Count of Songs with Non-Diatonic Chords by Year (Major)')
    axes[1].set_ylim(0, None)
    axes[1].set_xlim(df_non_diatonic_counts_major['Year'].min(), df_non_diatonic_counts_major['Year'].max())
    axes[1].set_xticks(df_non_diatonic_counts_major['Year'])
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylabel('N of Non-diatonic Chords')

    plt.tight_layout()
    plt.show()

