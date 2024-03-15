import numpy as np
import numpy as np
from scipy.fft import fft, fftfreq

def note_frequency(note, octave=4):
    # Mapping of note names to their semitone distances from A in the same octave
    note_semitones = {
        'C': -9, 'C#': -8, 'Db': -8, 'D': -7, 'D#': -6, 'Eb': -6,
        'E': -5, 'F': -4, 'F#': -3, 'Gb': -3, 'G': -2, 'G#': -1, 'Ab': -1,
        'A': 0, 'A#': 1, 'Bb': 1, 'B': 2
    }

    # Calculate the semitone distance from A4
    semitone_distance_from_A4 = note_semitones[note] + 12 * (octave - 4)

    # Frequency of A4
    A4_freq = 440

    # Calculate the frequency using the formula for equal temperament tuning
    frequency = A4_freq * (2 ** (semitone_distance_from_A4 / 12))

    return frequency


def find_harmonic_tension(frequencies, fs=44100, duration=1, n_samples=2000):
    """
    Plots any number of notes based on their frequencies, focusing on zero crossings from negative to positive,
    and identifies the region where these specific crossings are most closely in phase.

    :param frequencies: List of frequencies for the notes to be plotted.
    :param fs: Sampling frequency in Hz.
    :param duration: Duration of the signals in seconds.
    :param n_samples: Number of samples to show in the plot.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # Generate time array

    def find_zero_crossings_neg_to_pos(y):
        crossings = np.where(np.diff(np.sign(y)) > 0)[0]
        return crossings[crossings < n_samples]

    zero_crossing_times = []
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'grey', 'olive', 'cyan']
    combined_signal = np.zeros_like(t)

    for i, freq in enumerate(frequencies):
        note_signal = np.cos(2 * np.pi * freq * t)
        combined_signal += note_signal
        adj_zero_crossings = find_zero_crossings_neg_to_pos(note_signal)
        zero_crossing_times.append(t[adj_zero_crossings])



    # Combine and plot the overall signal
    adj_zero_crossings_combined = find_zero_crossings_neg_to_pos(combined_signal)


    # Find minimal delta t where specific crossings are closely in phase
    from itertools import product
    delta_ts = []
    alignment_times = []

    for zc_combination in product(*zero_crossing_times):
        max_time = max(zc_combination)
        min_time = min(zc_combination)
        delta_t = max_time - min_time
        delta_ts.append(delta_t)
        alignment_times.append((min_time, max_time))

    min_delta_t_index = np.argmin(delta_ts)
    min_delta_t = delta_ts[min_delta_t_index]
    alignment_time = alignment_times[min_delta_t_index]


    tension = alignment_time[1] - alignment_time[0]
    return tension,alignment_time,combined_signal

def find_combined_signal_frequency(combined_signal, fs=44100):
    """
    Computes the frequency spectrum of the combined signal and identifies the dominant frequency.

    :param combined_signal: The combined signal of the chord.
    :param fs: Sampling frequency in Hz.
    :return: Dominant frequency in the combined signal.
    """
    # Compute the FFT of the combined signal
    signal_fft = fft(combined_signal)
    signal_freq = fftfreq(len(combined_signal), 1/fs)

    # Compute magnitude spectrum and find the index of the maximum magnitude
    magnitude_spectrum = np.abs(signal_fft)
    dominant_freq_index = np.argmax(magnitude_spectrum[:len(combined_signal)//2])

    return signal_freq[dominant_freq_index]

def adjust_notes_octave(chord_notes):
    note_semitones = {
        'C': -9, 'C#': -8, 'Db': -8, 'D': -7, 'D#': -6, 'Eb': -6,
        'E': -5, 'F': -4, 'F#': -3, 'Gb': -3, 'G': -2, 'G#': -1, 'Ab': -1,
        'A': 0, 'A#': 1, 'Bb': 1, 'B': 2
    }
    adjusted_notes = [chord_notes[0] + '4']  # Assuming the root note is in the 4th octave
    for note in chord_notes[1:]:
        # Determine the semitone distance from the root note to adjust octave as needed
        root_semitone = note_semitones[chord_notes[0]]
        note_semitone = note_semitones[note]
        octave_adjustment = 4 + (1 if note_semitone < root_semitone else 0)
        adjusted_note = note + str(octave_adjustment)
        adjusted_notes.append(adjusted_note)
    return adjusted_notes

#calculates the stationary subharmonic tension (Δt) of a chord.


if __name__ == '__main__':
    chord_notes = [note_frequency("C",4), note_frequency("E",4), note_frequency("G",4)]
    tension = common_period = find_harmonic_tension(chord_notes)
    print(tension)
    print(f"The common subharmonic period for the C Major chord is: {common_period} seconds")