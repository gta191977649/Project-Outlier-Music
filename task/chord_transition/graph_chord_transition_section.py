import os
from model.song import Song
from plot.chord_transition_plot import *
from feature.analysis import *
import matplotlib.cm as cm

SECTION_COLOR = {
    "intro": "tab:purple",
    "verse": "blue",
    "solo": "tab:green",
    "bridge": "tab:brown",
    "end": "black",
    "outro": "tab:green",
    "chorus": "red",
    "inst": "tab:pink",
    "start": "tab:pink",
    "break": "tab:pink",
}

FREQ = {}


def find_section_label(time, s):
    for section in s:
        if section['start'] >= time < section['end']:
            return section['label']
    return False


def is_transition_in_patterns(a, b, patterns):
    a = a.replace(":", "")
    b = b.replace(":", "")
    for pat in patterns:
        for i in range(len(pat["pattern"]) - 1):
            chord_a = pat["pattern"][i].replace(":", "")
            chord_b = pat["pattern"][i + 1].replace(":", "")
            if a == chord_a and b == chord_b: return True
    return False


if __name__ == '__main__':
    TARGET_MODE = "minor"
    TARGET_SECTION = "solo"
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
    # plot
    plot = ChordTransitionPlot(f"Chord Transition\n{TARGET_SECTION} ({TARGET_MODE})", mode=TARGET_MODE)

    # pattern display
    # for song in song_collections:
    #     sections = song.section
    #     if not song.mode == TARGET_MODE: continue
    #     for pat in song.chord_pattern:
    #         for i in range(len(pat["pattern"])-1):
    #             chord_a = pat["pattern"][i]
    #             chord_b = pat["pattern"][i+1]
    #             chord_a = chord_a.replace(":","")
    #             chord_b = chord_b.replace(":","")
    #             # Non-diatonic to Non-diatonic
    #             if not is_diatonic_chord(chord_a, TARGET_MODE) and not is_diatonic_chord(chord_b, TARGET_MODE):
    #                 plot.addChordTransition(chord_a, chord_b, "red")
    #             # Diatonic to diatonic
    #             if is_diatonic_chord(chord_a, TARGET_MODE) and is_diatonic_chord(chord_b, TARGET_MODE):
    #                 plot.addChordTransition(chord_a, chord_b, "blue")
    #             # Diatonic to non-diatonic or vese versa
    #             if not is_diatonic_chord(chord_a, TARGET_MODE) and is_diatonic_chord(chord_b, TARGET_MODE):
    #                 plot.addChordTransition(chord_a, chord_b, "orange")
    #             if is_diatonic_chord(chord_a, TARGET_MODE) and not is_diatonic_chord(chord_b, TARGET_MODE):
    #                 plot.addChordTransition(chord_a, chord_b, "orange")

    # Visualize the transitions by section
    # Goal: understand the relation from non-diatonic transitions to music section (like which parts may have most non-diatonic)
    SECTION_FREQ = {}
    for song in song_collections:
        sections = song.section
        if not song.mode == TARGET_MODE: continue
        for i in range(len(song.chord_transposed) - 1):
            time_a, beat_a, chord_a = song.chord_transposed[i]
            time_b, beat_b, chord_b = song.chord_transposed[i + 1]


            chord_a = chord_a.replace(":", "")
            chord_b = chord_b.replace(":", "")
            if is_transition_in_patterns(chord_a, chord_b, song.chord_pattern):
                section_label = find_section_label(float(time_a), sections)
                if not section_label:
                    print("section is not fond!")
                    continue
                if section_label not in SECTION_FREQ:
                    SECTION_FREQ[section_label] = [0, 0]
                if is_diatonic_chord(chord_a, TARGET_MODE) and is_diatonic_chord(chord_b, TARGET_MODE):
                    SECTION_FREQ[section_label][0] += 1
                if not is_diatonic_chord(chord_a, TARGET_MODE) and not is_diatonic_chord(chord_b, TARGET_MODE):
                    SECTION_FREQ[section_label][1] += 1

                if section_label == TARGET_SECTION:
                    # Non-diatonic to Non-diatonic
                    if not is_diatonic_chord(chord_a, TARGET_MODE) and not is_diatonic_chord(chord_b, TARGET_MODE):
                        plot.addChordTransition(chord_a, chord_b, "red")
                    # Diatonic to diatonic
                    if is_diatonic_chord(chord_a, TARGET_MODE) and is_diatonic_chord(chord_b, TARGET_MODE):
                        plot.addChordTransition(chord_a, chord_b, "blue")
                    # Diatonic to non-diatonic or vese versa
                    if not is_diatonic_chord(chord_a, TARGET_MODE) and is_diatonic_chord(chord_b, TARGET_MODE):
                        plot.addChordTransition(chord_a, chord_b, "tab:orange")
                    if is_diatonic_chord(chord_a, TARGET_MODE) and not is_diatonic_chord(chord_b, TARGET_MODE):
                        plot.addChordTransition(chord_a, chord_b, "tab:orange")

    plot.showPlot()
    print(SECTION_FREQ)
    # print(SECTION_FREQ)
    # Sorting data for a consistent plot appearance
    sorted_data = sorted(SECTION_FREQ.items(), key=lambda x: sum(x[1]), reverse=True)
    categories = [item[0] for item in sorted_data]
    values1 = [item[1][0] for item in sorted_data]  # First set of values
    values2 = [item[1][1] for item in sorted_data]  # Second set of values

    # Preparing the index for the categories
    index = range(len(categories))

    # Setting up the plot
    fig, ax = plt.subplots(figsize=(6, 3))
    bar_width = 0.35

    # Plotting both sets of data
    bars1 = ax.bar(index, values1, bar_width, label='Diatonic Transitions')
    bars2 = ax.bar([p + bar_width for p in index], values2, bar_width, label='Non diatonic Transitions')

    # Adding some labels and title
    ax.set_xlabel('Section')
    ax.set_ylabel('Frequency')
    ax.set_title('Dual Bar Plot of Category Frequencies')
    ax.set_xticks([p + bar_width / 2 for p in index])
    ax.set_xticklabels(categories, rotation=0)
    ax.legend()

    plt.tight_layout()
    plt.show()
