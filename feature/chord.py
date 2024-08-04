from tslearn.metrics import dtw_path,dtw
from model.song import Song
from feature import extract as feature

import pandas as pd
import matplotlib.pyplot as plt

key_map = {
    "C": 0,
    "C#": 0.5,
    "Db": 0.5,
    "D": 1,
    "D#": 1.5,
    "Eb": 1.5,
    "E": 2,
    "F": 2.5,
    "F#": 3,
    "Gb": 3,
    "G": 3.5,
    "G#": 4,
    "Ab": 4,
    "A": 4.5,
    "A#": 5,
    "Bb": 5,
    "B": 5.5
}
# Initialize an empty reverse map
reverse_key_map = {}

# Populate the reverse map by iterating over the key_map
for chord, value in key_map.items():
    if value in reverse_key_map:
        reverse_key_map[value] += f"/{chord}"
    else:
        reverse_key_map[value] = chord

def extractChordNumeralValues(chord_array, mode="major"):
    val_array = []

    for chord in chord_array:
        # Initialize base to None
        base = None

        # Case 1: Chord label with ":" splitter (e.g., C:maj, E:min)
        if ":" in chord:
            base = chord.split(":")[0]

        # Case 2: Chord label without ":" splitter (e.g., Cmaj, Emin)
        else:
            # Find the first occurrence of "maj" or "min"
            maj_index = chord.find("maj")
            min_index = chord.find("min")

            if maj_index != -1:
                base = chord[:maj_index]
            elif min_index != -1:
                base = chord[:min_index]
            else:
                # If neither "maj" nor "min" is found, assume the entire string is the base
                base = chord

        # Ensure the base is stripped of any whitespace
        base = base.strip() if base else None

        if base and base in key_map:
            val_array.append(key_map[base])
        else:
            print(f"'{chord}' is not a valid chord or its base '{base}' is not in the key map")

    return val_array


def translateNumeralValuesToChords(numeral_values, mode="major"):
    chord_array = []

    for value in numeral_values:
        # Get the base chord from the reverse map
        base_chord = reverse_key_map.get(value, None)

        if base_chord:
            # Append chord with mode (e.g., C:maj, D:min)
            chord = f"{base_chord}"
            chord_array.append(chord)
        else:
            print(f"Numeral value '{value}' does not correspond to a known chord.")

    return chord_array
def extractCadencePatternFeature(chord_label_array):

    cadence_pattern = []



def extractChordSummrisation(song : Song,window =16,start_on_down_beat = True):
    if not song:
        return []
    print("Extracting Summary Chord Pattern ...")

    WINDOW = window  # Default 4 chords (16 beats in total)
    HOP_SIZE = 1
    matched_patterns = {}
    cost_threshold = 1

    chord_name_ls = []
    chord_theta_ls = []
    chord_start_ls = []
    chord_beat_ls = []
    for c in song.chord:
        time, beat, chord = c
        if chord == "N":
            continue
        angle = feature.getChordVectorsAngleFromChord(chord)
        chord_name_ls.append(chord)
        chord_theta_ls.append(angle)
        chord_start_ls.append(float(time))
        chord_beat_ls.append(float(beat))

    downbeat_indices = [i for i, beat in enumerate(chord_beat_ls) if beat == 1.0]

    for i in downbeat_indices:
        if i + WINDOW > len(chord_theta_ls):
            break  # Exit if the window exceeds the list length

        reference_signal = chord_theta_ls[i:i + WINDOW]
        pattern_key = tuple(chord_name_ls[i:i + WINDOW])

        if pattern_key not in matched_patterns:
            matched_patterns[pattern_key] = {
                'matches': 0,
                'match_details': []
            }

        for j in downbeat_indices:
            if j <= i or j + WINDOW > len(chord_theta_ls):
                continue  # Skip comparing the same window or exceeding list length

            search_signal = chord_theta_ls[j:j + WINDOW]
            path_length = len(reference_signal) + len(search_signal)
            cost = dtw(reference_signal, search_signal) / path_length

            if cost < cost_threshold:
                matched_patterns[pattern_key]['match_details'].append({
                    'start_search': j,
                    'end_search': j + WINDOW - 1
                })
                matched_patterns[pattern_key]['matches'] += 1

    output = []
    for pattern, details in matched_patterns.items():
        if details['matches'] > 1:  # Consider patterns with at least 2 matches
            output.append({
                "pattern": pattern,
                "matches": details['matches'],
                "match_details": details['match_details']
            })
    return output
def plotChordSummary(song :Song, summary):
    chord_name_ls = []
    chord_theta_ls = []
    chord_start_ls = []
    chord_beat_ls = []
    for c in song.chord:
        time, beat, chord = c
        x = feature.getChordVectorsAngleFromChord(chord)
        chord_name_ls.append(chord)
        chord_theta_ls.append(x)
        chord_start_ls.append(time)
        chord_beat_ls.append(beat)

    df = pd.DataFrame({
        'chord_name': chord_name_ls,
        'chord_theta': chord_theta_ls,
        'start': chord_start_ls,
        'beat': chord_beat_ls,
    })

    x_values = range(len(df))

    # Sort the summary list by 'matches' key in descending order
    summary_sorted = sorted(summary, key=lambda x: x['matches'], reverse=True)

    total_pattern_groups = len(summary_sorted)
    colormap = plt.cm.get_cmap('tab10', total_pattern_groups)
    fig, axs = plt.subplots(max(total_pattern_groups, 1), 1, figsize=(15, 2 * max(total_pattern_groups, 1)),
                            squeeze=False)
    axs = axs.flatten()

    for idx, pattern_details in enumerate(summary_sorted):
        color = colormap(idx % colormap.N)

        axs[idx].plot(x_values, df['chord_theta'], marker='o', alpha=0.5, color='grey', drawstyle='steps-post',
                      markersize=2, zorder=1)

        # Plot each matching segment from match_details
        for match in pattern_details['match_details']:
            search_start_idx = match['start_search']
            search_end_idx = match['end_search']
            axs[idx].plot(x_values[search_start_idx:search_end_idx + 1],
                          df['chord_theta'][search_start_idx:search_end_idx + 1], marker='o', linewidth=3, color=color,
                          drawstyle='steps-post', markersize=2, zorder=2, label='Match')

        axs[idx].set_title(f"Pattern: {pattern_details['pattern']} - {pattern_details['matches']} Matches")
        axs[idx].set_ylabel('Chord Position')
        axs[idx].set_xticks(x_values)
        axs[idx].set_xticklabels(df['beat'], rotation='vertical', fontsize=8)
        axs[idx].set_xlim(left=0, right=max(x_values))
        axs[idx].set_ylim(min(df['chord_theta']), max(df['chord_theta']))

    plt.tight_layout()
    #fig.suptitle(f"{song.title} @{song.key} {song.mode}", fontsize=16, fontweight='bold')
    #plt.subplots_adjust(top=0.90)
    plt.show()

if __name__ == '__main__':
    singal = extractChordNumeralValues(["C:maj","G:maj"])
    print(singal)