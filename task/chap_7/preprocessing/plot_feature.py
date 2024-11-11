import os
import feature.msd_getter as h5
import feature.msd_dataset as msd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def group_files_by_release(directory):
    releases = {}
    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            full_path = os.path.join(directory, filename)
            song = h5.open_h5_file_read(full_path)
            release = h5.get_release(song).decode("utf-8")
            if release not in releases:
                releases[release] = []
            releases[release].append(full_path)
    return releases

def loadArtistCollection(directory):
    collection = []
    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            full_path = os.path.join(directory, filename)
            #song = h5.open_h5_file_read(full_path)
            collection.append(full_path)
    return collection
if __name__ == '__main__':
    rl = "Original Album Classics"
    releases = group_files_by_release("/Users/nurupo/Desktop/dev/msd/blue_oyster/")
    print(releases)
    collection = loadArtistCollection("/Users/nurupo/Desktop/dev/msd/blue_oyster/")
    x = []
    y = []
    for path in collection:
        song = msd.openSong(path)
        a = msd.getFeature(song, feature="mfcc")
        b = msd.getFeature(song, feature="loudness")
        x.append(a)
        y.append(b)

    print(x)
    #plt.plot(y, label="")
    #plt.scatter(x,y=[range(0,len(x))])
    plt.scatter(y,x)
    plt.show()

