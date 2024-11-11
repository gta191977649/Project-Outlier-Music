import feature.msd_getter as h5
import numpy as np
import speechpy
import pandas as pd
def openSong(path):
    song = h5.open_h5_file_read(path)
    return song
def getFeature(song,feature):
    if feature == 'mfcc':
        mfcc = h5.get_segments_timbre(song)
        # Some magic normalization used from speaker identification
        mfcc = speechpy.processing.cmvn(mfcc.T, True)
        mfcc = np.array(mfcc).T
        # normalize mfcc for all 12 coffs
        mfcc_norm = np.mean(mfcc, axis=0)
        return mfcc_norm

    if feature == "loudness":
        loudness = h5.get_loudness(song)
        return loudness

    if feature == "tempo":
        tempo = h5.get_tempo(song)
        return tempo
    if feature == "year":
        year = h5.get_year(song)
        return year

    if feature =="energy":
        energy = h5.get_energy(song)
        return energy

    if feature =="danceability":
        danceability = h5.get_danceability(song)
        return danceability

def loadMSDCsvData(path):
    songs = []
    csv = pd.read_csv(path)
    for idx,item in csv.iterrows():
        songs.append(item.to_dict())
    return songs