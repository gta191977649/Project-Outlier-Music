from feature.dataset import *
from feature.chord import *
from plot.chord_transition_plot import *
from feature.analysis import *
import numpy as np

def loadSongsWithMeta(TARGET_MODE):
    #Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is
    # derived. Major is represented by 1 and minor is 0.
    TARGET_MODE = TARGET_MODE == "major" and 1 or 0
    PATH = "/Users/nurupo/Desktop/dev/music4all/"
    audio_folder = "audios"
    path_csv = os.path.join(PATH, "stratified_songs_pop.csv")

    song_collections = []
    if os.path.exists(path_csv):
        csv = pd.read_csv(path_csv)
        for _, item in csv.iterrows():
            id = item['id']
            mode = int(item['mode'])
            h5_path = os.path.join(PATH, audio_folder, id + ".h5")
            song = Song.from_h5(h5_path)
            if mode == TARGET_MODE:
                song_collections.append(song)
    return song_collections

def plotTrendsConsiderCandence(mode):
    # songs = loadSongCollection(r"/Users/nurupo/Desktop/dev/music4all/h5_pop_all", mode=mode)
    songs = loadSongsWithMeta(mode)
    # songs = loadSongCollection(r"/Users/nurupo/Desktop/dev/audio/aimyon", mode=mode)
    chord_signals = []
    chord_labels = []
    cadece_consider = {}
    cadece_consider["major"] = [
        ["G:maj", "C:maj"],  # Perfect Cadence
        ["F:maj", "C:maj"],  # Plagal Cadence
        ["C:maj", "G:maj"],  # Half Cadence
        ["D:maj", "G:maj"],  # Half Cadence
        ["F:maj", "G:maj"],  # Half Cadence
        ["G:maj", "A:min"],  # Deceptive Cadence
    ]

    # cadece_consider["minor"] = [
    #     ["E:min", "A:min"],  # Perfect Cadence
    #     ["D:min", "A:min"],  # Plagal Cadence
    #     ["A:min", "E:min"],  # Half Cadence
    #     ["B:dim", "E:min"],  # Half Cadence
    #     ["D:min", "E:min"],  # Half Cadence
    #     ["E:min", "F#:maj"],  # Deceptive Cadence
    # ]
    cadece_consider["minor"] = [
        ["E:maj", "A:min"],  # Perfect Cadence (V → i)
        ["D:min", "A:min"],  # Plagal Cadence (iv → i)
        ["A:min", "E:maj"],  # Half Cadence (i → V)
        ["B:dim", "E:maj"],  # Half Cadence (vii° → V)
        ["D:min", "E:maj"],  # Half Cadence (iv → V)
        ["E:maj", "F:maj"]  # Deceptive Cadence (V → VI)
    ]

    for target_song in songs:
        chords = target_song.extractChordProgressionLabels(transposed=True)
        x = extractChordNumeralValuesConsiderMode(chords)
        x = filterRepeatSignal(x)
        chords = filterRepeatSignal(chords)

        for cadece in cadece_consider[mode]:
            cadence_signal = extractChordNumeralValuesConsiderMode(cadece)
            matches = find_cadence_patterns(x, cadence_signal, min_preceding_chords=2)
            for start, end in matches:
                chord_signals.append(x[start:end])
                # label = convert_roman_label(chords[start:end], mode=mode)
                chord_labels.append(chords[start:end])
                # chord_labels.append(chords[start:end])
            # plotHarmonicProgression(chord_singal)

    # X_train = stretch_to_max_length(chord_singals)
    X_train = np.array(chord_labels)

    plot = ChordTransitionPlot(f"Chord Transition Graph - ({mode})", mode=mode)

    # Loop & add chords transitions
    for chord_progression in X_train:
        borrowed_keys = identify_borrowed_chords(chord_progression, mode)
        for i in range(len(chord_progression) - 1):
            a = chord_progression[i].replace(":", "")
            b = chord_progression[i + 1].replace(":", "")
            if a in borrowed_keys or b in borrowed_keys:
                plot.addChordTransition(a, b, "red")
            else:
                plot.addChordTransition(a, b, "blue")
    plot.showPlot()

def plotTrendsDirectly(TARGET_MODE):
    # just plot from raw pattern
    songs = loadSongsWithMeta(TARGET_MODE)
    plot = ChordTransitionPlot(f"Chord Transition Graph -({TARGET_MODE})", mode=TARGET_MODE)

    for song in songs:
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


if __name__ == '__main__':
    mode = "minor"
    #plotTrendsConsiderCandence(mode)
    plotTrendsDirectly(mode)
