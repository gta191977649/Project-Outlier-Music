import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def cross_correlation(signal, impulse_signal):
    impulse_len = len(impulse_signal)
    signal = (signal - np.mean(signal)) / (np.std(signal) * len(signal))
    impulse_signal = (impulse_signal - np.mean(impulse_signal)) / (np.std(impulse_signal))

    corr = np.correlate(signal, impulse_signal, mode='full')
    lags = np.arange(-impulse_len + 1, len(signal))

    return corr, lags

# Example signals
# I-I-IV-V Pattern
signal = np.array([1,1,4,5]*10)
impulse_signal = np.array([5,1]) # Perfect Cadence V -> I

# Compute cross-correlation
corr, lags = cross_correlation(signal, impulse_signal)

peaks, _ = find_peaks(corr)


# Highlight peaks on the signal
peak_indices = lags[peaks]

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(10, 6))

# Plot signal
axes[0].plot(signal, label='Signal',color='b',drawstyle="steps-post",)
axes[0].set_title('Harmonic Pattern')

# Highlight peaks on the signal plot
for peak in peak_indices:
    if 0 <= peak < len(signal):
        axes[0].plot(peak, signal[peak], 'ro', color='r')

# Plot impulse signal
axes[1].plot(impulse_signal, label='Cadence Impulse Signal',drawstyle="steps-mid", color='orange')
axes[1].set_title('Cadence Impulse Signal (V-I)')

# Plot cross-correlation with stem plot
axes[2].stem(lags, corr, label='Cross-correlation', linefmt='g')
axes[2].set_title('Cross-correlation')


# Highlight peaks on the cross-correlation plot
axes[2].plot(lags[peaks], corr[peaks], 'ro')

# Adjust layout
plt.tight_layout()
plt.show()
