import math

from feature.analysis import *
from feature.transition import *
from model.harmonic import *
from tslearn.metrics import dtw_path, dtw
from model.song import *
import os
import pandas as pd

def summaryChordPattern(chordsArray, WINDOW=16):
    if not chordsArray: return []
    #print("Extracting Summary Chord Pattern ...")

    START_ON_DOWNBEAT = True  # Set algorithm to only start search on chord that is on downbeat
    # WINDOW = 16  # measures for chord progession length
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
                    # print(cost)
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

    return output


def filterRepeatedChords(chord_progression):
    chord_progression_processed = []
    current = None
    for chord in chord_progression:
        if not chord == current:
            chord_progression_processed.append(chord)
            current = chord

    return chord_progression_processed

def evaluate_ending_cadence_score(chord_progression, mode="major"):
    ending_cadences_major = [
        ("V", "I", 4),  # "Authentic (Perfect) Cadence"
        ("IV", "I", 3),  # "Plagal Cadence"
        ("*", "V", 1),  # "Half Cadence"
        ("V", "vi", 2),  # "Deceptive (Interrupted) Cadence"
    ]

    ending_cadences_minor = [
        ("V", "i", 4),  # "Authentic (Perfect) Cadence",
        ("iv", "i", 3),  # "Plagal Cadence",
        ("*", "V", 1),  # "Half Cadence",
        ("V", "VI", 2)  # "Deceptive (Interrupted) Cadence",
    ]
    ending_cadences = mode == "major" and ending_cadences_major or ending_cadences_minor

    # process progression
    #print(chord_progression)
    chord_progression_processed = filterRepeatedChords(chord_progression)


    # detect ending cadences score
    overall_score = 0
    if len(chord_progression_processed) < 2:
        #print("Detect Chord progression is less 2 units!")
        return 0
    last_two_chords = tuple(chord_progression_processed[-2:])
    for cadence in ending_cadences:
        start_chord, end_chord, score = cadence
        # Check for Half Cadence which can have any start chord
        if start_chord == "*" and last_two_chords[1] == end_chord:
            overall_score += score
        # Check for exact match
        elif last_two_chords == (start_chord, end_chord):
            overall_score += score
    return overall_score


def evaluate_transition_resolve_score(chord_progression):
    #progression = filterRepeatedChords(chord_progression)
    progression = filterRepeatedChords(chord_progression)

    #print(progression)
    scores = []
    for i in range(len(progression)-1):
        s = compute_transition_resolution(progression[i], progression[i+1])
        if s < 0: #when detect reosolve
            scores.append(abs(s))

    scores = np.array(scores)
    return np.sum(scores)

def evaluate_ending_transition_resolve_score(chord_progression):
    progression = filterRepeatedChords(chord_progression)
    if len(progression) < 2:
        #print("Detect Chord progression is less 2 units!")
        return 0
    progression = progression[-2:]

    score = compute_transition_resolution(progression[0], progression[1])
    #print(progression,score)
    if score > 0: # means no reolve
        return 0

    return abs(score)

TOTAL_PROGRESSIONS = 0
FULL_PROGRESSIONS = 0
def evaluate_completeness_score(song):

    global FULL_PROGRESSIONS
    global TOTAL_PROGRESSIONS

    result = song.chord_pattern
    TOTAL_PROGRESSIONS += len(result)
    scores = []
    for pat in result:
        progression = pat["pattern"]
        c, n_diatonic, count_non_diatonic = anlysisromanMumerals(progression, song.mode == "major")
        cadence = evaluate_ending_cadence_score(c, song.mode)

        transition = evaluate_ending_transition_resolve_score(progression)
        #transition = evaluate_transition_resolve_score(progression)
        score = cadence + transition

        if score > 0:
            FULL_PROGRESSIONS +=1


        scores.append(score)

    scores = np.array(scores)
    return np.sum(scores)



if __name__ == '__main__':


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


    for idx,song in enumerate(song_collections):
        score = evaluate_completeness_score(song)
        print(f"REMAIN: {len(song_collections)-idx}")


    print(FULL_PROGRESSIONS,TOTAL_PROGRESSIONS)
