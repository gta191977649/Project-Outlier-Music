from feature.analysis import *
from model.song import *
from plot.chord_transition_plot import *
from pychord import Chord
import os
import pandas as pd
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

    plot = ChordTransitionPlot(f"Chord Transition Graph -({TARGET_MODE})", mode=TARGET_MODE)

    for song in song_collections:
        if not song.mode == TARGET_MODE: continue
        for ptn in song.chord_pattern:
            borrowed_keys = identify_borrowed_chords(ptn["pattern"], TARGET_MODE)
            for i in range(len(ptn["pattern"]) - 1):
                a = ptn["pattern"][i]
                b = ptn["pattern"][i + 1]
                if a in borrowed_keys or b in borrowed_keys:
                    plot.addChordTransition(a, b, "red")
                else:
                    plot.addChordTransition(a, b, "blue")

    plot.showPlot()
