import numpy as np
import matplotlib.pyplot as plt


def note_frequency(note, octave):
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

def plot_notes_in_phase(frequencies, fs=44100, duration=1, n_samples=2000):
    """
    Plots any number of notes based on their frequencies, focusing on zero crossings from negative to positive,
    and identifies the region where these specific crossings are most closely in phase.

    :param frequencies: List of frequencies for the notes to be plotted.
    :param fs: Sampling frequency in Hz.
    :param duration: Duration of the signals in seconds.
    :param n_samples: Number of samples to show in the plot.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # Generate time array
    plt.figure(figsize=(12, 6))

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

        # Plot note
        plt.plot(t[0:n_samples], note_signal[0:n_samples], label=f'Note {i + 1}',
                 color=colors[i % len(colors)])
        plt.scatter(t[adj_zero_crossings], note_signal[adj_zero_crossings], color=colors[i % len(colors)], s=10,
                    zorder=5)

    # Combine and plot the overall signal
    adj_zero_crossings_combined = find_zero_crossings_neg_to_pos(combined_signal)
    plt.plot(t[0:n_samples], combined_signal[0:n_samples], label='Combined Notes', linestyle='--',
             color='grey')
    plt.scatter(t[adj_zero_crossings_combined], combined_signal[adj_zero_crossings_combined], color='black', s=10,
                zorder=5)

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

    # Highlight minimal delta t region
    plt.axvspan(alignment_time[0], alignment_time[1], color='red', alpha=0.3,
                label='In phase region (tension dt)')
    tension = alignment_time[1] - alignment_time[0]
    plt.xlim(0, t[n_samples - 1])
    plt.title(f'In phase tension: {tension}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage with three notes C, E, G
chord_notes = [note_frequency("C",4), note_frequency("E",4), note_frequency("G",4)]

plot_notes_in_phase(chord_notes)

# Cmin Chord
#plot_notes_in_phase([261.63, 311.13 ,392.00])
