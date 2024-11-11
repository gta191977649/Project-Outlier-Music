from feature.dataset import *
from feature.chord import *
from plot.chord_transition_plot import *
from feature.analysis import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
def normalize_chord_labels(chord_array):
    # Define a dictionary to map equivalent chords to a unique representation
    chord_mapping = {
        "Db:maj": "C#:maj",
        "Eb:maj": "D#:maj",
        "Fb:maj": "E:maj",
        "Gb:maj": "F#:maj",
        "Ab:maj": "G#:maj",
        "Bb:maj": "A#:maj",
        "Cb:maj": "B:maj",
        "Db:min": "C#:min",
        "Eb:min": "D#:min",
        "Fb:min": "E:min",
        "Gb:min": "F#:min",
        "Ab:min": "G#:min",
        "Bb:min": "A#:min",
        "Cb:min": "B:min"
    }

    # Create a new array for the normalized chords
    normalized_chord_array = np.array([chord_mapping.get(chord, chord) for chord in chord_array])

    return normalized_chord_array


def map_chords_to_circle_of_fifths(chord_array, chord_order):
    # Create a mapping from chord label to its position in the circle of fifths
    chord_to_index = {chord: i for i, chord in enumerate(chord_order)}

    # Replace each chord label in the array with its corresponding index
    mapped_array = np.array([chord_to_index.get(chord, -1) for chord in chord_array])

    return mapped_array


def plot_parallel_coordinates(X_train_0, X_train_1, X_train_2, X_train_3,mode):
    # Create a DataFrame with the chord data and a label for positions
    df = pd.DataFrame({
        '1': X_train_0,
        '2': X_train_1,
        '3': X_train_2,
        '4': X_train_3
    })

    # Add a column for color groups if needed (optional)
    df['Group'] = ['chord_' + str(i) for i in range(len(X_train_0))]

    # Plot parallel coordinates using pandas built-in function
    plt.figure(figsize=(7, 5))
    pd.plotting.parallel_coordinates(df, 'Group', color="b", linewidth=1, alpha=0.1)
    plt.legend([], [], frameon=False)

    # Set the y-ticks to match chord labels from 1 to 12 with 0.5 intervals
    y_ticks = np.arange(1, 13, 0.5)
    y_labels = [number_to_chord_label(y) for y in y_ticks]  # Generate chord labels for y-axis

    # Set y-ticks and labels
    plt.yticks(ticks=y_ticks, labels=y_labels)

    # Set axis labels and title
    plt.title(f'Chords (Positions 1 to 4) - {mode}')
    plt.ylabel('Chord Label')
    plt.xlabel('Chord Position')
    plt.xticks(rotation=0)

    # Set the y-axis limits (ylim)
    #plt.ylim(1, 12.5)  # Set the y-axis range from 1 to 12.5

    # Show the plot
    plt.show()

def plot_parallel_coordinates_with_label(X_train_0, X_train_1, X_train_2, X_train_3,mode):
    # Create a DataFrame with the chord data and a label for positions
    df = pd.DataFrame({
        '1': X_train_0,
        '2': X_train_1,
        '3': X_train_2,
        '4': X_train_3
    })
    # number_to_chord_label
    # Add a column for color groups if needed (optional)
    df['Group'] = ['chord_' + str(i) for i in range(len(X_train_0))]

    # Plot parallel coordinates using pandas built-in function
    plt.figure(figsize=(7, 5))
    pd.plotting.parallel_coordinates(df, 'Group', color="b", linewidth=1, alpha=0.01,sort_labels=True)
    plt.legend([], [], frameon=False)
    # Set axis labels and title
    plt.title(f'Chords (Positions 1 to 4) - {mode}')
    plt.ylabel('Chord Label')
    plt.xlabel('Chord Position')
    plt.xticks(rotation=0)



    # Show the plot
    plt.show()

def loadPopSongsWithMeta(TARGET_MODE):

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
            if song.mode == TARGET_MODE:
                song_collections.append(song)
    return song_collections

def plotTrendsConsiderCandence(mode):
    # songs = loadSongCollection(r"/Users/nurupo/Desktop/dev/music4all/h5_pop_all", mode=mode)
    #songs = loadSongsWithMeta(mode)
    songs = loadSongCollection(r"/Users/nurupo/Desktop/dev/audio/akb48", mode="major")

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
        ["B:min", "E:maj"],  # Half Cadence (vii° → V)
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
                # chord_labels.append(label)
                chord_labels.append(chords[start:end])
            # plotHarmonicProgression(chord_singal)

    # X_train = stretch_to_max_length(chord_singals)
    # X_train = np.array(chord_labels)  # The Tokenization processing is done in ChordProgressionAnalyzer
    X_train = np.array(chord_signals)  # The Tokenization processing is done in ChordProgressionAnalyzer

    X_train_0 = X_train.T[0]
    X_train_1 = X_train.T[1]
    X_train_2 = X_train.T[2]
    X_train_3 = X_train.T[3]

    print("processed, plot now...")
    plot_parallel_coordinates(X_train_0, X_train_1, X_train_2, X_train_3, mode)

    print("OK")

def plotTrendsDirectly(TARGET_MODE):
    songs = loadSongsWithMeta(TARGET_MODE)

    chord_labels = []
    for song in songs:
        if not song.mode == TARGET_MODE: continue
        for ptn in song.chord_pattern:
            borrowed_keys = identify_borrowed_chords(ptn["pattern"], TARGET_MODE)
            #print(ptn["pattern"])
            for i in range(len(ptn["pattern"]) - 1):
                # 16 beats per bar, in this case we only want chord from downbeat
                progression = []
                progression.append(ptn["pattern"][0])
                progression.append(ptn["pattern"][4])
                progression.append(ptn["pattern"][8])
                progression.append(ptn["pattern"][12])
                chord_labels.append(extractChordNumeralValuesConsiderMode(progression))

    X_train = np.array(chord_labels)
    X_train_0 = X_train.T[0]
    X_train_1 = X_train.T[1]
    X_train_2 = X_train.T[2]
    X_train_3 = X_train.T[3]

    print("processed, plot now...")
    plot_parallel_coordinates(X_train_0, X_train_1, X_train_2, X_train_3, mode)

    print("OK")


if __name__ == '__main__':
    mode = "minor"
    plotTrendsConsiderCandence(mode)
    #plotTrendsDirectly(mode)

