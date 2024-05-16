import os
import pandas as pd
from model.song import Song
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration for year intervals
YEAR_INTERVAL = 1


def process_song(song, non_diatonic_counts_by_year, total_counts_by_year, year_interval):
    year = int(song.release)
    # Group the year into intervals
    year_group = (year // year_interval) * year_interval

    # Increment total song count for year group
    total_counts_by_year[year_group] = total_counts_by_year.get(year_group, 0) + 1

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
    total_counts_by_year_minor = {}
    total_counts_by_year_major = {}

    if os.path.exists(path_csv):
        csv = pd.read_csv(path_csv)
        for _, item in csv.iterrows():
            id = item['id']
            h5_path = os.path.join(PATH, audio_folder, id + ".h5")
            song = Song.from_h5(h5_path)

            if song.mode == "minor":
                process_song(song, non_diatonic_counts_by_year_minor, total_counts_by_year_minor, YEAR_INTERVAL)
            elif song.mode == "major":
                process_song(song, non_diatonic_counts_by_year_major, total_counts_by_year_major, YEAR_INTERVAL)

    # Calculate proportion of non-diatonic songs
    proportion_minor = {year: non_diatonic_counts_by_year_minor[year] / total_counts_by_year_minor.get(year, 1) for year
                        in non_diatonic_counts_by_year_minor}
    proportion_major = {year: non_diatonic_counts_by_year_major[year] / total_counts_by_year_major.get(year, 1) for year
                        in non_diatonic_counts_by_year_major}

    # Convert dictionaries to DataFrame and sort by Year
    df_proportion_minor = pd.DataFrame(list(proportion_minor.items()),
                                       columns=['Year', 'Proportion_of_Non_Diatonic_Songs']).sort_values(by='Year')
    df_proportion_major = pd.DataFrame(list(proportion_major.items()),
                                       columns=['Year', 'Proportion_of_Non_Diatonic_Songs']).sort_values(by='Year')

    # # plot
    # fig, axes = plt.subplots(2, 1, figsize=(5, 5))
    #
    # # Plot for 'minor' mode
    # sns.barplot(x='Year', y='Proportion_of_Non_Diatonic_Songs', data=df_proportion_minor, ax=axes[0], color="red" ,edgecolor="black",linewidth=2)
    # sns.regplot(x=np.arange(len(df_proportion_minor)), y='Proportion_of_Non_Diatonic_Songs', data=df_proportion_minor,
    #             ax=axes[0],
    #             scatter=False, color="tab:red", order=3, ci=None)
    # axes[0].set_title('Proportion of Songs with Non-Diatonic Chords by Year Interval (Minor)')
    # axes[0].set_xticklabels(df_proportion_minor['Year'], rotation=0)
    # axes[0].set_xlabel('Year (Interval 5)')
    # axes[0].set_ylabel('Proportion of Non-diatonic Songs')
    #
    # # Plot for 'major' mode
    # sns.barplot(x='Year', y='Proportion_of_Non_Diatonic_Songs', data=df_proportion_major, ax=axes[1], color="blue",edgecolor="black",linewidth=2)
    # sns.regplot(x=np.arange(len(df_proportion_major)), y='Proportion_of_Non_Diatonic_Songs', data=df_proportion_major,
    #             ax=axes[1],
    #             scatter=False, color="tab:blue", order=3, ci=None)
    # axes[1].set_title('Proportion of Non-Diatonic Songs vs Year (Major)')
    # axes[1].set_xticklabels(df_proportion_major['Year'], rotation=0)
    # axes[1].set_xlabel('Year (Interval 5)')
    # axes[1].set_ylabel('Proportion of Non-diatonic Songs')
    #
    # plt.tight_layout()
    # plt.style.use('classic')
    # plt.show()

    # Combine minor and major proportions into one DataFrame
    df_proportion_minor['Mode'] = 'Minor'
    df_proportion_major['Mode'] = 'Major'
    combined_df = pd.concat([df_proportion_minor, df_proportion_major])

    # Plot
    plt.figure(figsize=(8, 3))
    sns.barplot(x='Year', y='Proportion_of_Non_Diatonic_Songs', hue='Mode', data=combined_df,
                palette=[ 'lightgrey','lightcoral'],edgecolor="black",linewidth=1)
    plt.title('Proportion of Songs with Non-Diatonic Chords by Year Interval')
    plt.xticks(rotation=90)
    plt.xlabel('Year')
    plt.ylabel('Proportion')
    plt.tight_layout()
    plt.style.use('classic')
    plt.show()