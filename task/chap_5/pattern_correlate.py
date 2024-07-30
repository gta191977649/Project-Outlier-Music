from model.song import Song
from feature.chord import *

import numpy as np
from scipy.signal import correlate,correlation_lags

def filterRepeatSignal(signal):
    current = None
    out = []
    for x in signal:
        if not x == current:
            out.append(x)
            current = x
    return out


def cross_correlation(x, y):
    """Compute the cross-correlation of x and y manually."""
    len_x = len(x)
    len_y = len(y)
    result = np.zeros(len_x + len_y - 1)

    # Pad the sequences with zeros to perform the cross-correlation
    padded_x = np.pad(x, (len_y - 1, len_y - 1), 'constant')

    # Compute the cross-correlation
    for lag in range(-(len_y - 1), len_x):
        if lag < 0:
            result[lag + len_y - 1] = np.sum(y[:lag] * padded_x[len_y - 1 + lag:len_y - 1])
        else:
            result[lag + len_y - 1] = np.sum(y * padded_x[lag:lag + len_y])

    return result




if __name__ == '__main__':
    file = r"F:\music4all\pop_h5\0A5hn2Tk34p1g4ZU.h5"
    song = Song.from_h5(file)


    labels = song.extractChordProgressionLabels()
    #signal = extractChordNumeralValues(labels)

    #signal = filterRepeatSignal(signal)
    signal = [0, 4, 5, 4, 1.5, 3.5]
    #perfect cadence V → I
    # cadence_signal = extractChordNumeralValues([
    #     "G:maj",
    #     "C:maj",
    # ])

    #signal = [0, 4, 5, 4, 1.5, 3.5, 0, 1.5]
    cadence_signal = [5, 4, 1.5]

    # Compute cross-correlation using scipy.signal.correlate
    corr = correlate(signal,cadence_signal, mode='full',method="direct")
    # Calculate the lags for the correlation
    lags = correlation_lags(len(signal),len(cadence_signal),  mode="full")

    print("Signal:", signal)
    print("Cadence Signal:", cadence_signal)
    print("Correlation:", corr)
    print("Lags:", lags)
    print("L_S:", len(signal))
    print("L_C:", len(cadence_signal))
    print("COR_L:", len(corr))
    print("Max Sim Index:", np.argmax(corr))
    print("Max Sim Lag:", lags[np.argmax(corr)])

    # Plotting
    plt.figure(figsize=(10, 5))

    plt.subplot(3, 1, 1)
    plt.plot(range(len(signal)), signal, label='Signal', color='blue')
    plt.title('Signal')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(range(len(cadence_signal)), cadence_signal, label='Cadence Signal', color='red')
    plt.title('Cadence Signal')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.stem( corr, label='Correlation', basefmt=" ")
    plt.title('Correlation with Lags')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.legend()

    plt.tight_layout()
    plt.show()