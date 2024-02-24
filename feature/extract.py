import time

import madmom, scipy.stats, numpy as np
from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label
from madmom.features.chords import DeepChromaChordRecognitionProcessor, CRFChordRecognitionProcessor, \
    CNNChordFeatureProcessor
from pychord import Chord
from pychord.constants import NOTE_VAL_DICT
from model.note import *
from model.vectormodel import Chord as VectorModel
import time

INDEX_NOTE_DICT = {v: k for k, v in NOTE_VAL_DICT.items()}

def map_vector(chord_21,name):
    map_21 = {'A': eNote.A,
              'D': eNote.D,
              'G': eNote.G,
              'C': eNote.C,
              'F': eNote.F,
              'Bb': eNote.Bb,
              'A#': eNote.Bb,
              'Eb': eNote.Eb,
              'D#': eNote.Eb,
              'G#': eNote.Ab,
              'Ab': eNote.Ab,
              'C#': eNote.Db,
              'Db': eNote.Db,
              'F#': eNote.Fsharp,
              'Gb': eNote.Fsharp,
              'B': eNote.B,
              'E': eNote.E}
    temp = []
    for i in chord_21:
        temp.append(map_21[i])

    return VectorModel(temp,name=name)


def getChordVectorsAngleFromChords(chordsArray):
    angles = []
    for c in chordsArray:
        time, beat, chord = c
        if chord == "N":  # Skip None Chord
            new_chord = (time, beat, "N")
            angles.append(0)
            continue
        chord = chord.replace(":", "")
        c = Chord(chord)
        notes = c.components()
        angle = map_vector(notes,chord)
        angles.append(angle)
    return angles
def getChordVectorsAngleFromChord(chord):
    if chord == "N" or chord == "None":  # Skip None Chord
        return 0
    chord = chord.replace(":", "")
    c = Chord(chord)
    notes = c.components()
    angle = map_vector(notes, chord).temp_theta
    return angle
def extractBeatAlignedChordLabels(file):
    print("Extract Beat Aligned Chord ...")
    t_s = time.time()
    # detect chord
    dcp = DeepChromaProcessor()
    decode = DeepChromaChordRecognitionProcessor()
    chroma = dcp(file)
    chords = decode(chroma)
    # detect beats
    beat_processor = RNNDownBeatProcessor()
    beat_decoder = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)
    beats = beat_decoder(beat_processor(file))
    # get beat align chord
    chordsArray = []
    chord_idx = 0
    for beat_idx in range(len(beats) - 1):
        curr_beat_time, curr_beat = beats[beat_idx]
        # find the corresponding chord for this beat
        while chord_idx < len(chords):
            chord_time, _, _ = chords[chord_idx]
            prev_beat_time, _ = (0, 0) if beat_idx == 0 else beats[beat_idx - 1]
            eps = (curr_beat_time - prev_beat_time) / 2
            if chord_time > curr_beat_time + eps:
                break
            chord_idx += 1

        # append to array
        _, _, prev_chord = chords[chord_idx - 1]
        chord = (curr_beat_time, curr_beat, prev_chord)

        chordsArray.append(chord)
    print(f"✅Beat Aligned Chord: {time.time() - t_s}")
    return chordsArray
def transposeBeatAlignedChordLabels(chordsArray, transpose_amount, target_scale="C"):
    print("Tranpose Beat Aligned Chord ...")
    transpoed_chords = []
    for c in chordsArray:
        time, beat, chord = c
        if chord == "N":  # Skip None Chord
            new_chord = (time, beat, "N")
            transpoed_chords.append(new_chord)
            continue
        chord = chord.replace(":", "")

        transposed_chord = Chord(chord)
        transposed_chord.transpose(transpose_amount)
        new_chord = (time, beat, transposed_chord)
        transpoed_chords.append(new_chord)
    return transpoed_chords
def calculate_transpose_amount(original_key, original_mode):
    """
     Calculate the amount needed to transpose from the original key and mode
     to the standardized key (C for major, A for minor).

     :param original_key: The starting key of the piece.
     :param original_mode: The mode of the piece ('major' or 'minor').
     :return: The transposition amount in semitones.
     """
    # Define target keys for Major (C) and Minor (A)
    target_major = "C"
    target_minor = "A"

    # Ensure the original key is properly capitalized to match the dictionary
    original_key = original_key.capitalize()

    # Get semitone value for original key and target keys
    original_key_val = NOTE_VAL_DICT.get(original_key)
    target_major_val = NOTE_VAL_DICT[target_major]
    target_minor_val = NOTE_VAL_DICT[target_minor]

    if original_key_val is None:
        raise ValueError(f"Original key '{original_key}' is not valid.")

    # Calculate transpose amount based on mode
    if original_mode.lower() == "major":
        transpose_amount = target_major_val - original_key_val
    elif original_mode.lower() == "minor":
        transpose_amount = target_minor_val - original_key_val
    else:
        raise ValueError("Mode should be 'major' or 'minor'.")

    # Normalize the transpose amount to the range [-6, 6] for minimal movement
    if transpose_amount > 6:
        transpose_amount -= 12
    elif transpose_amount < -6:
        transpose_amount += 12

    return transpose_amount
