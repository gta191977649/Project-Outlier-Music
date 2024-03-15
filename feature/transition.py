from feature.harmonic import *
from pychord import Chord
def compute_transition_resolution(chord_a_label,chord_b_label):
    N = 9
    c = 1

    # Frequencies for each notes
    chord_name_p = chord_a_label.replace(":","")
    chord_name_s = chord_b_label.replace(":","")

    chord_p = Chord(chord_name_p)
    chord_s = Chord(chord_name_s)

    chord_p_length = len(chord_p.components())
    chord_s_length = len(chord_s.components())
    # Get fundamental frequency of two chords notes
    adjusted_chord_p_notes = adjust_notes_octave(chord_p.components())
    adjusted_chord_s_notes = adjust_notes_octave(chord_s.components())
    # Get frequencies for adjusted chord notes
    chord_freq_p = [note_frequency(note[:-1], int(note[-1])) for note in adjusted_chord_p_notes]
    chord_freq_s = [note_frequency(note[:-1], int(note[-1])) for note in adjusted_chord_s_notes]

    # print(chord_freq_p)
    # print("====")
    # print(chord_freq_s)
    # chord_freq_p = [note_frequency(note, 4) for note in chord_p.components()]
    # chord_freq_s = [note_frequency(note, 4) for note in chord_s.components()]


    # Compute all subharmonic for two chords
    chord_harmonic_p = np.zeros((N, chord_p_length))
    chord_harmonic_s = np.zeros((N, chord_s_length))

    for note_index, freq in enumerate(chord_freq_p):
        for i in range(0, N):
            chord_harmonic_p[i, note_index] = freq / (i + 1)

    for note_index, freq in enumerate(chord_freq_s):
        for i in range(0, N):
            chord_harmonic_s[i, note_index] = freq / (i + 1)

    # print(chord_harmonic_p)
    # print(chord_harmonic_s)
    sum_terms = 0
    for j in range(N):
        # Analyze tensions and frequencies for each chord's j-th subharmonic
        chord_p_tension, _, chord_p_signal = find_harmonic_tension(chord_harmonic_p[j, :])
        chord_s_tension, _, chord_s_signal = find_harmonic_tension(chord_harmonic_s[j, :])
        chord_p_freq = find_combined_signal_frequency(chord_p_signal, fs=44100)
        chord_s_freq = find_combined_signal_frequency(chord_s_signal, fs=44100)

        ddt = chord_p_tension - chord_s_tension
        T_sub = 1 / chord_s_freq
        dt_s = chord_s_tension

        if dt_s < (1 / 2) * T_sub:  # Condition to include the interaction in the calculation
            ddt_hat = ddt / T_sub
            sum_terms += (ddt_hat / (dt_s * T_sub)) ** c
    # Final calculation of the overall measure of Transitional Harmony
    if N > 0:  # Prevent division by zero
        overall_transitional_harmony = (sum_terms / N) ** (1 / c)
    else:
        overall_transitional_harmony = 0
    return overall_transitional_harmony

if __name__ == '__main__':
    #progression = ["Fmaj", "Cmaj"]
    progression = ["Cmaj", "Dmin"]

    overall = 0
    for i in range(0,len(progression)-1):
        a = progression[i]
        b = progression[i+1]
        score = compute_transition_resolution(a,b)
        print(a,b,score)
        overall += score

    print(overall)