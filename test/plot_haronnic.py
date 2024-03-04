import numpy as np
import matplotlib.pyplot as plt

# Sampling parameters
fs = 44100  # Sampling frequency in Hz
duration = 1  # Duration in seconds
t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # Generate time array

# Frequencies for C, E, and G notes
f_c = 261.63  # Frequency of C4 in Hz
f_e = 329.63  # Frequency of E4 in Hz
f_g = 392.00  # Frequency of G4 in Hz

# Generate cosine waves for C, E, and G
cosine_c = np.cos(2 * np.pi * f_c * t)
cosine_e = np.cos(2 * np.pi * f_e * t)
cosine_g = np.cos(2 * np.pi * f_g * t)
chord_signal_cos = cosine_c + cosine_e + cosine_g  # Combined C major chord signal

def find_zero_crossings(y):
    return np.where(np.diff(np.sign(y)))[0]

# Find zero crossing points for each signal
zero_crossings_c = find_zero_crossings(cosine_c)
zero_crossings_e = find_zero_crossings(cosine_e)
zero_crossings_g = find_zero_crossings(cosine_g)
zero_crossings_chord = find_zero_crossings(chord_signal_cos)

# Adjust zero_crossings to within the first 1000 samples
def adjust_zero_crossings(zero_crossings):
    return zero_crossings[zero_crossings < 1000]

# Adjusted zero crossings for each signal
adj_zero_crossings_c = adjust_zero_crossings(zero_crossings_c)
adj_zero_crossings_e = adjust_zero_crossings(zero_crossings_e)
adj_zero_crossings_g = adjust_zero_crossings(zero_crossings_g)
adj_zero_crossings_chord = adjust_zero_crossings(zero_crossings_chord)

# Plot the signals with marked zero crossings, but only show the first 1000 samples
plt.figure(figsize=(12, 6))

# C Note
plt.plot(t[0:1000], cosine_c[0:1000], label='C Note (Cosine)',color='red',)
plt.scatter(t[adj_zero_crossings_c], cosine_c[adj_zero_crossings_c], color='red', s=10, zorder=5, label='Zero Crossings C')

# E Note
plt.plot(t[0:1000], cosine_e[0:1000], label='E Note (Cosine)',color='green',)
plt.scatter(t[adj_zero_crossings_e], cosine_e[adj_zero_crossings_e], color='green', s=10, zorder=5, label='Zero Crossings E')

# G Note
plt.plot(t[0:1000], cosine_g[0:1000], label='G Note (Cosine)', color='blue')
plt.scatter(t[adj_zero_crossings_g], cosine_g[adj_zero_crossings_g], color='blue', s=10, zorder=5, label='Zero Crossings G')

# Combined C Major Chord
plt.plot(t[0:1000], chord_signal_cos[0:1000], label='Combined C Major Chord (Cosine)', linestyle='--', color='grey')
plt.scatter(t[adj_zero_crossings_chord], chord_signal_cos[adj_zero_crossings_chord], color='grey', s=10, zorder=5, label='Zero Crossings Chord')

plt.xlim(0,t[1000])
plt.title('Cosine Waves and Zero Crossings for C, E, G Notes and Combined Chord (First 1000 Samples)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()
