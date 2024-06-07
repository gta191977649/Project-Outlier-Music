import madmom, scipy.stats, numpy as np
from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label
from madmom.features.chords import DeepChromaChordRecognitionProcessor, CRFChordRecognitionProcessor, \
    CNNChordFeatureProcessor
from feature.pattern import extractTontalPitchDistancePattern,computeTPSD,extractChromaticPattern


if __name__ == '__main__':
    #I, V, vi, IV

    # Test for some shits
    home = "C:maj"
    chords = [
        "C:maj",
        "D:min",
        "E:min",
        "F:maj",
        "G:maj",
        "A:min",
        "B:dim",
    ]
    pattern = extractTontalPitchDistancePattern(chords,key=home,mode="profile")

    print(pattern)
    home = "D:maj"
    chords = [
        "D:maj",
        "E:min",
        "F#:min",
        "G:maj",
        "A:maj",
        "B:min",
        "C#:dim",
    ]
    pattern = extractTontalPitchDistancePattern(chords, key=home, mode="profile")
    print(pattern)


    # Take on me chords
    home = "A:maj"

    # CHORD CHART
    # |A . . . ||C#m . . . ||F# . . . ||D . . . |
    #   take       on         me     ~~~~
    chords = [
        "A:maj",
        "C#:min",
        "F#:min",
        "D:maj",
    ]
    pattern = extractTontalPitchDistancePattern(chords, key=home, mode="profile")
    print(pattern)

    # She will be loved
    home = "C:maj"

    chords = [
        "C:maj",
        "G:maj",
        "A:min",
        "F:maj",
    ]

    pattern = extractTontalPitchDistancePattern(chords, key=home, mode="profile")
    print(pattern)

    # africa

    home = "F#:min"

    chords = [
        "F#:min",
        "D:maj",
        "A:maj",
        "E:maj",
    ]
    pattern = extractTontalPitchDistancePattern(chords, key=home, mode="profile")
    print(pattern)

    # 君はメロディー
    home = "Db:maj"

    chords = [
        "Db:maj",
        "Eb:maj",
        "C:min",
        "F:min",
    ]

    pattern = extractTontalPitchDistancePattern(chords, key=home, mode="profile")
    print(pattern)