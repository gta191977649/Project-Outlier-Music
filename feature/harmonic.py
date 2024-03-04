import numpy as np
import math
import matplotlib.pyplot as plt

# def lcm(a, b):
#     """Calculate the least common multiple of two numbers."""
#     return abs(a * b) // math.gcd(int(a), int(b))

def multiples(x,time):
    n = []
    for i in range(1,time):
        n.append(x*i)
    return np.array(n)

def get_note_frequency(note_label,tune = 440.0):
    A4_freq = tune
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note_position = notes.index(note_label) - 9
    frequency = A4_freq * (2 ** (note_position / 12))
    return frequency

def get_common_subharmonic_period(chord):
    # test
    f_c = get_note_frequency("C")
    f_e = get_note_frequency("E")
    f_g = get_note_frequency("G")

    # compute periods
    t_c = 1/f_c
    t_e = 1/f_e
    t_g = 1/f_g

    # find multiples of each notes
    m_c = multiples(t_c,10)
    m_e = multiples(t_e,10)
    m_g = multiples(t_g,10)

    print(m_c,m_e,m_g)

def find_common_subharmonic_periods(chord_notes, tune=440.0):
    Scale = 10000
    frequencies = [get_note_frequency(note, tune) for note in chord_notes]
    # Calculate periods from the given frequencies
    periods = [1 / freq for freq in frequencies]
    print(periods)
    # Convert periods to scaled integers
    scaled_periods = [int(period * Scale) for period in periods]

    # Calculate the LCM of scaled periods
    common_period_scaled = np.lcm.reduce(scaled_periods)

    # Convert the common period back to its original scale
    common_period = common_period_scaled / Scale

    return common_period

if __name__ == '__main__':
    chord_notes = ["C", "E", "G"]
    common_period = find_common_subharmonic_periods(chord_notes)
    print(f"The common subharmonic period for the C Major chord is: {common_period} seconds")