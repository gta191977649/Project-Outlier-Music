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

    song_collections = []

    if os.path.exists(path_csv):
        csv = pd.read_csv(path_csv)
        for _, item in csv.iterrows():
            id = item['id']
            h5_path = os.path.join(PATH, audio_folder, id + ".h5")
            song = Song.from_h5(h5_path)
            song_collections.append(song)

    borrowed_keys_by_year = {}


    for year in range(1980,2020):
        if year not in borrowed_keys_by_year:
            borrowed_keys_by_year[year] = {
                "n_non_diatonic": 0,
                "n_tonic":0,
                "n_chord":0,
                "n_borrowed_chord": 0,
                "normalized_borrowed_chord":0
            }
        for song in song_collections:
            if song.release == year:
                # count tonic chords
                for c in song.chord_transposed:
                    time,beat,chord = c
                    c = chord.replace(":","")
                    borrowed_keys_by_year[year]["n_chord"] += 1
                    if TARGET_MODE == "major":
                        if c == "Cmaj": borrowed_keys_by_year[year]["n_tonic"] += 1
                    else:
                        if c == "Amin": borrowed_keys_by_year[year]["n_tonic"] += 1

                for ptn in song.chord_pattern:
                    borrowed_keys = identify_borrowed_chords(ptn["pattern"], TARGET_MODE)
                    borrowed_keys_by_year[year]["n_non_diatonic"] += ptn["non_diatonic_chords_count"]


                    for i in range(len(ptn["pattern"]) - 1):
                        a = ptn["pattern"][i]
                        b = ptn["pattern"][i + 1]
                        if a in borrowed_keys or b in borrowed_keys:
                            borrowed_keys_by_year[year]["n_borrowed_chord"] += 1 * ptn["matches"]

    # normalize borrow chords
    for year in borrowed_keys_by_year:
        borrowed_keys_by_year[year]["normalized_borrowed_chord"] = borrowed_keys_by_year[year]["n_borrowed_chord"] / borrowed_keys_by_year[year]["n_non_diatonic"]
        borrowed_keys_by_year[year]["n_tonic"] = borrowed_keys_by_year[year]["n_tonic"] / borrowed_keys_by_year[year]["n_chord"]

    data = []
    for year, stats in borrowed_keys_by_year.items():
        data.append({
            "Year": year,
            "N-Tonic Chord": stats["n_tonic"],
            "Normalized Borrowed Chords": stats["n_borrowed_chord"],
            "Non-Diatonic Chords": stats["n_non_diatonic"],
            "normalized_borrowed_chord": stats["normalized_borrowed_chord"],
        })

    df = pd.DataFrame(data)

    df = df.sort_values(by="Year")

    df.to_csv("/Users/nurupo/Desktop/dev/music4all/borrowed_keys_by_year.csv",index=False)

    years = sorted(borrowed_keys_by_year.keys())

    borrowed_chords_counts = [borrowed_keys_by_year[year]["n_tonic"] for year in years]

    plt.figure(figsize=(15, 7))  # Set the figure size for better readability
    plt.bar(years, borrowed_chords_counts, color='blue')

    plt.xlabel('Year')
    plt.ylabel('Number of Borrowed Chords')
    plt.title('Number of Borrowed Chords by Year')
    plt.xticks(years, rotation='vertical')  # Rotate years for better visibility
    plt.grid(axis='y', linestyle='--')

    # Show the plot
    plt.tight_layout()  # Adjust the layout to make sure everything fits without overlap
    plt.show()