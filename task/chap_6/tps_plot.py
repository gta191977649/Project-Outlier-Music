import os
import numpy as np
import feature.extract as extract
from model.song import Song
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA,LatentDirichletAllocation
import matplotlib.cm as cm
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.metrics import dtw_path, dtw
from feature.format import format_chord_progression
from metric.tpsd.tps_comparison import TpsComparison
from feature.pattern import extractTontalPitchDistancePattern,computeTPSD,extractChromaticPattern
import scipy
import librosa
from scipy.ndimage import filters


if __name__ == '__main__':
    H5 = "/Users/nurupo/Desktop/dev/music4all/custom/MV君はメロディー Short ver.  AKB48[公式].h5"
    AUDIO = "/Users/nurupo/Desktop/dev/music4all/custom/MV君はメロディー Short ver.  AKB48[公式].mp3"

    #song = Song.from_h5(H5)
    song = Song(id="AKB",title="TEST", artist="TEST",file=AUDIO)

    chords = []
    for item in song.chord:
        time, beat, chord = item
        chords.append(chord)
    key = f"{song.key}:{song.mode[:3]}"

    tps = extractTontalPitchDistancePattern(chords, mode="profile", key=key)

    for i in range(len(song.chord)-1):
        time, beat, chord = song.chord[i]
        print(f"{time},{beat},{chord},{tps[i]}")

    plt.figure(figsize=(15, 3))
    plt.step(range(len(tps)), tps, where='mid', color='b', linewidth=2)
    # for peak in f_picks:
    #     plt.axvline(x=peak, color='red', linestyle='dotted', linewidth=1,lw=2)
    plt.xlim([0, len(tps)])
    plt.show()