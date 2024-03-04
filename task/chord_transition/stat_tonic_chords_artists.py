from feature.analysis import *
from model.song import *
from plot.chord_transition_plot import *
from pychord import Chord
import os
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    TARGET_MODE = "major"

    PATH = "/Users/nurupo/Desktop/dev/music4all/"
    audio_folder = "audios"
    path_csv = os.path.join(PATH, "stratified_songs_pop.csv")

    song_collections_by_artist = {}

    if os.path.exists(path_csv):
        csv = pd.read_csv(path_csv)
        for _, item in csv.iterrows():
            id = item['id']
            popularity = item['popularity']
            h5_path = os.path.join(PATH, audio_folder, id + ".h5")
            song = Song.from_h5(h5_path)
            artist = song.artist
            if artist not in song_collections_by_artist:
                song_collections_by_artist[artist] = []
            song_collections_by_artist[artist].append(song)

    stat_by_artist = {}
    for artist in song_collections_by_artist:
        if not artist in stat_by_artist:
            stat_by_artist[artist] = {
                "total_songs": 0,
                "n_tonic":0,
                "n_chords":0,
                "n_nondiatonic":0
            }
        for song in song_collections_by_artist[artist]:
            for c in song.chord_transposed:
                time, beat, chord = c
                chord = chord.replace(":","")
                if chord == "N": continue# skip empty chord
                stat_by_artist[artist]["n_chords"] += 1
                if song.mode == "major":
                    if chord == "Cmaj": stat_by_artist[artist]["n_tonic"] += 1
                else:
                    if chord == "Amin": stat_by_artist[artist]["n_tonic"] += 1

            stat_by_artist[artist]["total_songs"] += 1


    # normalize it
    for artist in stat_by_artist:
        if stat_by_artist[artist]["n_chords"] == 0:
            stat_by_artist[artist]["n_tonic"] = 0
            print(artist)
            continue
        stat_by_artist[artist]["n_tonic"] =  stat_by_artist[artist]["n_tonic"] / stat_by_artist[artist]["n_chords"]

    N = 30

    # Sort artists by the number of tonic chords and select the top N
    sorted_artists = sorted(stat_by_artist.items(), key=lambda x: x[1]['n_tonic'], reverse=True)[:N]

    # Extract artist names and their corresponding n_tonic values
    artist_names = [artist[0] for artist in stat_by_artist.items()][:N]
    n_tonic_values = [artist[1]['n_tonic'] for artist in stat_by_artist.items()][:N]

    # Plotting
    plt.figure(figsize=(10, 8))  # Set figure size
    plt.barh(artist_names, n_tonic_values, color='skyblue')  # Create horizontal bars
    plt.xlabel('Number of Tonic Chords')
    plt.ylabel('Artist')
    plt.title('Number of Tonic Chords by Artist')
    plt.xticks(rotation='vertical')  # Rotate x labels for better visibility if needed
    plt.grid(axis='x', linestyle='--')

    # Show plot
    plt.tight_layout()  # Adjust layout
    plt.show()
    #print(stat_by_artist)

