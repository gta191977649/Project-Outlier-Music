import madmom, scipy.stats, numpy as np
from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label
from madmom.features.chords import DeepChromaChordRecognitionProcessor, CRFChordRecognitionProcessor, \
    CNNChordFeatureProcessor
from feature.pattern import extractTontalPitchDistancePattern,computeTPSD,extractChromaticPattern
from metric.tpsd.tps_comparison import TpsComparison



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

    # Test for linear
    print("aa")
    # Test for some shits
    home = "C:maj"
    chords = [
        "C:maj",
        "D:min",
        "F:maj",
    ]
    pattern = extractTontalPitchDistancePattern(chords, key=home, mode="profile")
    print(pattern)
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
    pattern = extractTontalPitchDistancePattern(chords, key=home, mode="profile")
    print(pattern)

    # tps_comparison = TpsComparison(chord_a='C:maj', key_a='C:maj', chord_b='A:min', key_b='C:maj')
    # circle_of_fifth_rule = tps_comparison.circle_fifth_rule()
    # chord_distance_rule = tps_comparison.chord_distance_rule()
    #
    # tpsd_distance = tps_comparison.chord_distance_rule() / 2
    # tps_comparison.plot()
    # print(circle_of_fifth_rule)
    # print(chord_distance_rule)
    # print(tpsd_distance)