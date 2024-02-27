import os
import pandas as pd
from model.song import Song
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == '__main__':
    PATH = "/Users/nurupo/Desktop/dev/music4all/"
    audio_folder = "audios"
    path_csv = os.path.join(PATH, "stratified_songs_pop.csv")

    song_collections = []

    if os.path.exists(path_csv):
        csv = pd.read_csv(path_csv)
        for _, item in csv.iterrows():
            id = item['id']
            h5_path = os.path.join(PATH, audio_folder, id + ".h5")
            song = Song.from_h5(h5_path)
            song_collections.append(song)

    non_diatonic_by_key_major = {}
    non_diatonic_by_key_minor = {}

    total_song_by_key_major = {}

    for song in song_collections:



    for song in song_collections:
        key = song.key
        mode = song.mode
        year = int(song.release)
        stat_group = non_diatonic_by_key_major if mode == "major" else non_diatonic_by_key_minor

        #stat_group = mode == "major" and non_diatonic_by_key_major or non_diatonic_by_key_minor
        if key not in stat_group:
            stat_group[key] = 0

        if len(song.chord_pattern) > 0:
            for ptn in song.chord_pattern:
                count = ptn.get("non_diatonic_chords_count", 0)
                if count > 0:
                    stat_group[key] += 1
                    print(f"{song.title} has non diatonic chords!")
                    continue

    # Creating the subplots
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))

    # Major keys histogram
    ax[0].bar(non_diatonic_by_key_major.keys(), non_diatonic_by_key_major.values(), color='tab:blue')
    ax[0].set_title('Non-Diatonic Songs in Major Keys')
    ax[0].set_xlabel('Key Class')
    ax[0].set_ylabel('Number of Songs')

    # Minor keys histogram
    ax[1].bar(non_diatonic_by_key_minor.keys(), non_diatonic_by_key_minor.values(), color='tab:red')
    ax[1].set_title('Non-Diatonic Songs in Minor Keys')
    ax[1].set_xlabel('Key Class')
    ax[1].set_ylabel('Number of Songs')

    plt.tight_layout()
    plt.show()