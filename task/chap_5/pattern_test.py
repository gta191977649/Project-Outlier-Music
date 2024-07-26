import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def stretch_to_max_length(X_train):
    # Find the maximum length of the sequences in X_train
    max_length = max(len(seq) for seq in X_train)

    # Create a new array to hold the stretched sequences
    stretched_X_train = np.zeros((len(X_train), max_length))

    # Stretch each sequence to the max length using interpolation
    for i, seq in enumerate(X_train):
        original_indices = np.linspace(0, 1, len(seq))
        stretched_indices = np.linspace(0, 1, max_length)
        interpolator = interp1d(original_indices, seq, kind='linear')
        stretched_X_train[i] = interpolator(stretched_indices)

    return stretched_X_train

# Generate test signals: simple sine waves of different frequencies
def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    frequencies = x * freq
    y = np.sin((2 * np.pi) * frequencies)
    return y

# Parameters
sample_rate = 1000  # Samples per second
duration = 1.0  # Seconds

# Generate test signals
signal1 = generate_sine_wave(5, sample_rate, duration)  # 5 Hz sine wave
signal2 = generate_sine_wave(10, sample_rate, duration)  # 10 Hz sine wave
signal3 = generate_sine_wave(20, sample_rate, duration)  # 20 Hz sine wave

# Combine signals into a list
X_train = [signal1, signal2, signal3]

# Stretch the signals to the max length
stretched_X_train = stretch_to_max_length(X_train)

# Plotting the original and stretched signals
plt.figure(figsize=(12, 10))

# Plot original signals
for i, seq in enumerate(X_train):
    plt.subplot(2, 1, 1)
    plt.plot(seq, label=f'Signal {i+1}')
    plt.title('Original Signals')
    plt.legend()

# Plot stretched signals
for i, seq in enumerate(stretched_X_train):
    plt.subplot(2, 1, 2)
    plt.plot(seq, label=f'Stretched Signal {i+1}')
    plt.title('Stretched Signals')
    plt.legend()

plt.tight_layout()
plt.show()
