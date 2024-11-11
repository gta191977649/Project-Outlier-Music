from tslearn.metrics import dtw_path,dtw
from model.song import Song
from feature import extract as feature

import pandas as pd
import matplotlib.pyplot as plt

# key_map = {
#     "C": 1,
#     "C#": 1.5,
#     "Db": 1.5,
#     "D": 2,
#     "D#": 2.5,
#     "Eb": 2.5,
#     "E": 3,
#     "F": 4,
#     "F#": 4.5,
#     "Gb": 4.5,
#     "G": 5,
#     "G#": 5.5,
#     "Ab": 5.5,
#     "A": 6,
#     "A#": 6.5,
#     "Bb": 6.5,
#     "B": 7
# }
key_map = {
    "C": 1,
    "C#": 2,
    "Db": 2,
    "D": 3,
    "D#": 4,
    "Eb": 4,
    "E": 5,
    "F": 6,
    "F#": 7,
    "Gb": 7,
    "G": 8,
    "G#": 9,
    "Ab": 9,
    "A": 10,
    "A#": 11,
    "Bb": 11,
    "B": 12
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

def extractChordNumeralValuesConsiderMode(chord_array):
    val_array = []

    for chord in chord_array:
        # Initialize base and mode flags
        base = None
        is_minor = False

        # Check if chord contains ':'
        if ":" in chord:
            base, mode = chord.split(":")
            base = base.strip()  # Ensure no extra spaces in the base
            if mode.strip() == "min":
                is_minor = True
        else:
            # If no ':', check for "maj" or "min" directly in the chord label
            if "min" in chord:
                base = chord.split("min")[0].strip()
                is_minor = True
            elif "maj" in chord:
                base = chord.split("maj")[0].strip()
            else:
                base = chord.strip()  # Treat as base with no mode information

        # Map base to key_map and apply minor adjustment if needed
        if base and base in key_map:
            value = key_map[base] + (0.5 if is_minor else 0)
            val_array.append(value)
        else:
            print(f"'{chord}' is not a valid chord or its base '{base}' is not in the key map")

    return val_array


def number_to_chord_label(number):
    base_value = int(number) % 12  # Use modulo 12 to handle wrapping
    base_value = 12 if base_value == 0 else base_value  # Correct for B -> 12
    is_minor = (number % 1 == 0.5)

    # Get the base chord from reverse_key_map
    chord_label = reverse_key_map.get(base_value, "Invalid")
    mode = "min" if is_minor else "maj"
    return f"{chord_label}"

# Function to translate numeral values to chord labels considering minor/major
def translateNumeralValuesToChords(numeral_values):
    chord_array = []

    for value in numeral_values:
        # Determine if the chord is minor or major based on .5
        is_minor = (value % 1) == 0.5
        base_value = int(value)  # Get the integer part for the base chord

        # Get the base chord from the reverse map
        base_chord = reverse_key_map.get(base_value % 12, None)  # Modulo to handle wrapping

        if base_chord:
            # Append the chord with appropriate mode (min or maj)
            mode = "min" if is_minor else "maj"
            chord = f"{base_chord}:{mode}"
            chord_array.append(chord)
        else:
            print(f"Numeral value '{value}' does not correspond to a known chord.")

    return chord_array
def convert_roman_label(chord_sequence, mode):
    # Define the diatonic chords and their Roman numerals for C major and C minor
    diatonic_chords = {
        "major": {
            "C:maj": "I", "D:min": "ii", "E:min": "iii", "F:maj": "IV",
            "G:maj": "V", "A:min": "vi", "B:dim": "vii°"
        },
        "minor": {
            "C:min": "i", "D:dim": "ii°", "Eb:maj": "III", "F:min": "iv",
            "G:min": "v", "Ab:maj": "VI", "Bb:maj": "VII"
        }
    }

    # Define the possible secondary dominants
    secondary_dominants = {
        "major": {
            "D:maj": "V/V", "E:maj": "V/vi", "A:maj": "V/ii",
            "B:maj": "V/iii", "C:maj": "V/IV"
        },
        "minor": {
            "C:maj": "V/iv", "D:maj": "V/v", "Eb:maj": "V/VI",
            "F:maj": "V/VII", "G:maj": "V/III"
        }
    }

    roman_numerals = []
    for i, chord in enumerate(chord_sequence):
        # Check if the chord is a potential secondary dominant
        if chord in secondary_dominants[mode]:
            # Look ahead to see if the next chord matches the secondary dominant's target
            if i + 1 < len(chord_sequence):
                next_chord = chord_sequence[i + 1]
                target_chord = secondary_dominants[mode][chord].split('/')[1]
                if next_chord in diatonic_chords[mode] and diatonic_chords[mode][next_chord].lower() == target_chord:
                    roman_numerals.append(secondary_dominants[mode][chord])
                    continue

        # If not a secondary dominant, use the diatonic chord label
        if chord in diatonic_chords[mode]:
            roman_numerals.append(diatonic_chords[mode][chord])
        else:
            # For any non-diatonic chords, use a simple representation
            root, quality = chord.split(':')
            roman_numerals.append(f"{root}({quality})")

    return roman_numerals


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


def find_cadence_patterns(main_signal, cadence_pattern, min_preceding_chords=2, allow_repetitions=True):
    """
    Find multiple occurrences of a cadence pattern in the main signal using exact matching,
    with an option to allow or disallow repetitive chords.

    Parameters:
    - main_signal: The main chord progression signal (list of numbers)
    - cadence_pattern: The cadence pattern to search for (list of numbers)
    - min_preceding_chords: Minimum number of chords required before the cadence pattern (default: 2)
    - allow_repetitions: Whether to allow repetitive chords in the progression (default: True)

    Returns:
    - A list of tuples, each containing (start_index, end_index) of found patterns
    """
    pattern_length = len(cadence_pattern)
    matches = []

    for i in range(min_preceding_chords, len(main_signal) - pattern_length + 1):
        # Check if the cadence pattern matches exactly
        if main_signal[i:i + pattern_length] == cadence_pattern:
            # Check if there are enough preceding chords
            if i >= min_preceding_chords:
                # If repetitions are not allowed, check for unique chords
                if not allow_repetitions:
                    progression = main_signal[i - min_preceding_chords:i + pattern_length]
                    if len(set(progression)) == len(progression):
                        matches.append((i - min_preceding_chords, i + pattern_length))
                else:
                    matches.append((i - min_preceding_chords, i + pattern_length))

    return matches


if __name__ == '__main__':
    singal = extractChordNumeralValues(["C:maj","G:maj"])
    print(singal)