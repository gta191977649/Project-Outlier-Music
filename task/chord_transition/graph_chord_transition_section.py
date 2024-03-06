import os
from model.song import Song
from plot.chord_transition_plot import *

SECTION_COLOR = {
    "intro":"tab:purple",
    "verse":"blue",
    "solo":"tab:orange",
    "bridge":"tab:brown",
    "end":"black",
    "outro":"tab:pink",
    "chorus":"red",
    "inst":"tab:pink",
    "start":"tab:pink",
    "break":"tab:pink",
}

FREQ = {}
def find_section_label(time,s):
    for section in s:
        if section['start'] >= time < section['end']:
            return section['label']
    return None


if __name__ == '__main__':
    TARGET_MODE = "major"
    PATH = "/Users/nurupo/Desktop/dev/music4all/europe/"
    # loop all folder
    song_collections = []
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith(".h5"):
                filepath = os.path.join(root, file)
                filename = os.path.splitext(file)[0]
                song = Song.from_h5(filepath)
                if song.mode == TARGET_MODE:
                    song_collections.append(song)
    #plot
    plot = ChordTransitionPlot(f"Chord Transition Graph -({TARGET_MODE})", mode=TARGET_MODE)

    for song in song_collections:
        sections = song.section
        if not song.mode == TARGET_MODE: continue
        for i in range(len(song.chord_transposed)-1):
            time_a,beat_a,chord_a = song.chord_transposed[i]
            time_b,beat_b,chord_b = song.chord_transposed[i+1]
            if not chord_a == "N" and not chord_b == "N":
                label = find_section_label(float(time_b),sections)
                if not label: continue
                if not label in FREQ:
                    FREQ[label] = 0
                #print(time_a,sections[0]["label"])
                if label == "bridge":
                    plot.addChordTransition(chord_a, chord_b, SECTION_COLOR[label])
                FREQ[label] +=1
    print(FREQ)
    plot.showPlot()