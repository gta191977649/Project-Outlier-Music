from feature import extract as feature
from model.harmonic import *
from tslearn.metrics import dtw_path,dtw
from model.song import *
from pychord import Chord
import matplotlib.pyplot as plt

def summaryChordPattern(chordsArray,WINDOW = 16):
    if not chordsArray: return []
    print("Extracting Summary Chord Pattern ...")

    START_ON_DOWNBEAT = True  # Set algorithm to only start search on chord that is on downbeat
    #WINDOW = 16  # measures for chord progession length
    HOP_SIZE = 1  # Hop size of 1 allows overlapping windows
    matched_patterns = {}
    used_patterns = set()  # Set to keep track of unique patterns already used in a match
    cost_threshold = 1
    matched_indices = set()  # Set to keep track of indices that have been matched

    # START
    chord_name_ls = []
    chord_theta_ls = []
    chord_start_ls = []
    chord_beat_ls = []
    # Loop through Chords
    for c in chordsArray:
        time, beat, chord = c
        if chord == "N": continue
        angle = feature.getChordVectorsAngleFromChord(chord)
        chord_name_ls.append(chord)
        chord_theta_ls.append(angle)
        chord_start_ls.append(float(time))
        chord_beat_ls.append(float(beat))
    i = 0
    while i < len(chord_theta_ls) - WINDOW + 1:
        reference_signal = chord_theta_ls[i:i + WINDOW]
        pattern_key = tuple(chord_name_ls[i:i + WINDOW])  # Convert the pattern to a tuple to use it as a dictionary key
        # Make sure we only start from downbeats
        if START_ON_DOWNBEAT:
            if not chord_beat_ls[i] == 1.0:
                i += HOP_SIZE
                continue

        if i in matched_indices:  # Skip if this index is part of a matched pattern
            i += HOP_SIZE
            continue
        if pattern_key not in matched_patterns:
            matched_patterns[pattern_key] = {
                'start_ref': i,
                'end_ref': i + WINDOW - 1,
                'matches': []
            }

            # Use a while loop for dynamic control over the index j
            j = i + WINDOW
            while j < len(chord_theta_ls) - WINDOW + 1:
                if j in matched_indices:  # Skip if this index is part of a matched pattern
                    j += HOP_SIZE
                    continue
                if START_ON_DOWNBEAT:
                    if not chord_beat_ls[j] == 1.0:
                        j += HOP_SIZE
                        continue
                search_signal = chord_theta_ls[j:j + WINDOW]
                path_length = len(reference_signal) + len(search_signal)

                cost = dtw(reference_signal, search_signal)
                # normalize cost
                cost = cost / path_length

                if cost < cost_threshold:
                    print(cost)
                    matched_patterns[pattern_key]['matches'].append({
                        'start_search': j,
                        'end_search': j + WINDOW - 1
                    })
                    matched_indices.update(range(j, j + WINDOW))  # Mark these indices as matched
                    j += WINDOW  # Skip current found chord position
                else:
                    j += HOP_SIZE  # Move to the next position

            # If the current pattern has one or more matches, mark the pattern as used
            if len(matched_patterns[pattern_key]['matches']) > 0:
                matched_indices.update(range(i, i + WINDOW))  # Mark these indices as matched
                i += WINDOW
                continue

        i += HOP_SIZE  # Move to the next position

    # Print All Found Matches
    output = []
    for pattern, details in matched_patterns.items():
        if len(details['matches']) > 0:
            output.append({
                "pattern": pattern,
                "matches": len(details['matches']),
            })
           # print(pattern,len(details['matches']))
    #pattern_freq = {pattern: len(details['matches']) for pattern, details in matched_patterns.items() if len(details['matches']) > 0}
    #plot
    # Plotting the signal
    x_values = range(len(chord_theta_ls))
    filtered_patterns = {pattern: details for pattern, details in
                         sorted(matched_patterns.items(), key=lambda item: len(item[1]['matches']), reverse=True) if
                         len(details['matches']) > 0}

    total_pattern_groups = len(filtered_patterns)

    colormap = plt.cm.get_cmap('tab10', total_pattern_groups)  # 'tab10' has a wide range of distinct colors

    color_idx = 0

    fig, axs = plt.subplots(total_pattern_groups, 1, figsize=(15, 2 * total_pattern_groups), squeeze=True)
    axs = axs.flatten()

    # Plot each unique pattern and its matches in a separate subplot
    for (pattern, details), ax in zip(filtered_patterns.items(), axs):
        # Get color from the colormap, use modulo to cycle through colors if there are more patterns than colors
        color = colormap(color_idx % colormap.N)
        color_idx += 1  # Increment color index for the next distinct pattern

        # Plot the entire signal in a neutral color in the background
        neutral_color = 'grey'
        ax.plot(x_values, chord_theta_ls, marker='o', alpha=0.5, color=neutral_color, drawstyle='steps-post',
                markersize=2, zorder=1)

        # Reference pattern segment
        ref_start_idx = details['start_ref']
        ref_end_idx = details['end_ref']
        ax.plot(x_values[ref_start_idx:ref_end_idx + 1],chord_theta_ls[ref_start_idx:ref_end_idx + 1], marker='o',
                color=color, drawstyle='steps-post', markersize=2, zorder=2, label='Reference')

        # Plot each matching segment
        for match in details['matches']:
            search_start_idx = match['start_search']
            search_end_idx = match['end_search']
            ax.plot(x_values[search_start_idx:search_end_idx + 1],
                    chord_theta_ls[search_start_idx:search_end_idx + 1],
                    marker='o', linewidth=3, color=color, drawstyle='steps-post', markersize=2, zorder=2, label='Match')

        # Set title and labels for each subplot
        ax.set_title(f"Pattern: {pattern} - {len(details['matches'])} Matches")
        ax.set_ylabel('Angle')
        ax.set_xticks(x_values)
        ax.set_xticklabels(chord_beat_ls, rotation='vertical', fontsize=8)
        ax.set_xlim(left=0, right=max(x_values))
        ax.set_ylim(min(chord_theta_ls), max(chord_theta_ls))
        # ax.legend()

    # Finalizing the plot
    plt.tight_layout()
    plt.show()


    return output


def detect_ending_cadence(chord_progression):
    ending_cadences_major = {
        ["V","I"]: 4,   # "Authentic (Perfect) Cadence"
        ["IV","I"]:3,   #"Plagal Cadence"
        ["*","V"]: 1,   #"Half Cadence"
        ["V","vi"]:2,   #"Deceptive (Interrupted) Cadence"
    }

    ending_cadences_minor = {
        ["V", "i"]: 4,  #"Authentic (Perfect) Cadence",
        ["iv", "i"]:3,  #"Plagal Cadence",
        ["*", "V"]: 1,  #"Half Cadence",
        ["V", "VI"]:2,  #"Deceptive (Interrupted) Cadence",
    }


if __name__ == '__main__':
    song = Song.from_h5('/Users/nurupo/Desktop/dev/music4all/europe/The Final Countdown [NNiTxUEnmKI].h5')

    result = summaryChordPattern(song.chord_transposed,WINDOW=16)
    print("_________")
    overall = 0
    for pat in result:
        progression = pat["pattern"]
        roman_label = pat["roman_label"]
        tension = 0
        for chord in progression:
            harmonic = Harmonic(chord)
            #print(harmonic.tension*1000)
            tension += harmonic.tension*1000
        overall = overall + tension
        print(tension)
    print(f"overall:{overall}")
    #print(result)
    #c = Chord("Cmaj")
    #print(c.components())
