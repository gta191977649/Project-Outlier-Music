from model.song import *
from feature.extract import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    song = Song.from_h5("/Users/nurupo/Desktop/dev/music4all/bk/17さいのうた  ユイカMV.h5")
    signal = getChordVectorsAngleFromChords(song.chord_transposed)
    signal = signal[:300]
    plt.figure(figsize=(10, 2))
    plt.step(range(len(signal)), signal, where='mid', color='black', linewidth=1.5)

    #plt.xlabel('Time')
    #plt.ylabel('Angle (radians)')
    plt.xlim(left=0, right=len(signal) - 1)  # Correct way to set x-axis limits
    plt.ylim(min(signal), max(signal))  # Adjust y-axis limits based on the signal

    plt.grid(False)
    plt.tight_layout()
    plt.show()
