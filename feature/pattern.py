from feature import extract as feature
from tslearn.metrics import dtw_path,dtw
import matplotlib.pyplot as plt
from metric.tpsd.tps_comparison import TpsComparison
def to_harte_label(chord):
    # Check if the chord is already in Harte format
    if any(suffix in chord for suffix in [':maj', ':min', ':dim', ':aug']):
        return chord
    # Replace chord suffixes with Harte format
    chord = chord.replace('maj', ':maj').replace('min', ':min').replace('dim', ':dim').replace('aug', ':aug')
    return chord

# This methods implements de Berardinis et al.(2023)
# The Harmonic Memory: a Knowledge Graph of harmonic paterns as a trustworthy framework for computational creativity
def extractTontalPitchDistancePattern(chordsArray,key='C:maj',mode="offset"):
    if not chordsArray: return []
    tpsd_singal = []
    if mode == "offset":
        for i in range(0, len(chordsArray) - 1):
            if chordsArray[i] == 'N' or chordsArray[i + 1] == 'N': continue
            a = to_harte_label(chordsArray[i])
            b = to_harte_label(chordsArray[i + 1])
            key = to_harte_label(key)
            print(a,b)
            tpd = TpsComparison(chord_a=a, chord_b=b, key_a=key, key_b=key)
            tpsd_singal.append(tpd.get_tpsd_distance())

    if mode == "profile":
        for i in range(0, len(chordsArray)):
            if chordsArray[i] == 'N' or key == 'N': continue
            a = to_harte_label(chordsArray[i])
            b = to_harte_label(key)
            key = to_harte_label(key)

            tpd = TpsComparison(chord_a=a, chord_b=b, key_a=key, key_b=key)
            tpsd_singal.append(tpd.chord_distance_rule()/2)
            #tpsd_singal.append(tpd.distance())

    return tpsd_singal

# implements Hua Cui Kan's 色彩和声 Vector Model
def extractChromaticPattern(chordsArray):
    if not chordsArray: return []
    chromatic_singal = []
    for i in range(0, len(chordsArray)):
        if chordsArray[i] == "N":
            chromatic_singal.append(0)
            continue

        angle = feature.getChordVectorsAngleFromChord(chordsArray[i])
        chromatic_singal.append(angle)
    return chromatic_singal
def computeTPSD(a,b,key='C:maj'):
    a = to_harte_label(a)
    b = to_harte_label(b)
    tpsd = TpsComparison(chord_a=a, chord_b=b, key_a=key, key_b=key)
    return tpsd.get_tpsd_distance()
def extractChangeChordPattern(chordsArray):
    if not chordsArray: return []
    current_name = None
    chord_vaild_ls = []
    chord_time_ls = []
    chord_beat_ls = []
    for c in chordsArray:
        time, beat, chord = c
        if chord == "N" or chord == None: continue
        if chord == current_name: continue

        chord_vaild_ls.append(chord)
        chord_time_ls.append(time)
        chord_beat_ls.append(time)

        current_name = chord
    chord_sequence = {
        'valid_sequence':chord_vaild_ls,
        'time':chord_time_ls,
        "beat":chord_beat_ls
    }
    return chord_sequence

def summaryChordPattern(chordsArray,window = 16):
    if not chordsArray: return []
    print("Extracting Summary Chord Pattern ...")

    START_ON_DOWNBEAT = True  # Set algorithm to only start search on chord that is on downbeat
    WINDOW = window  # measures for chord progession length
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
                    #print(cost)
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
    return output
