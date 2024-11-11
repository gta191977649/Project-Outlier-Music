import os
import feature.msd_getter as h5
import matplotlib.pyplot as plt
import speechpy as speechpy
import numpy as np
from matplotlib.cm import get_cmap
import feature.msd_dataset as msd
from sklearn.ensemble import RandomForestRegressor

def group_files_by_release(directory):
    releases = {}
    # Iterate over all .h5 files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            full_path = os.path.join(directory, filename)
            # Open the .h5 file for reading using your custom library
            song = h5.open_h5_file_read(full_path)
            release = h5.get_release(song).decode("utf-8")


            if release not in releases:
                releases[release] = []
            releases[release].append(os.path.join(directory,filename))

    return releases

if __name__ == '__main__':
    # song = msd.openSong("/Users/nurupo/Desktop/dev/msd/blue_oyster/TRCNEOX128F92C2C13.h5")
    # mfcc = msd.getFeature(song,feature="mfcc")
    rl = 'Secret Treaties'
    releases = group_files_by_release("/Users/nurupo/Desktop/dev/msd/blue_oyster/")
    print(releases)

    mfcc_coff = []
    for path in releases[rl]:
        song = msd.openSong(path)
        mfcc = msd.getFeature(song,feature="mfcc")

        mfcc_coff.append(mfcc)

    # Convert list to numpy array for easier manipulation
    mfcc_coff = np.array(mfcc_coff)  # Shape: (num_songs, num_coefficients, num_frames)

    # Calculate mean and standard deviation across songs for each coefficient
    mean_mfcc = np.mean(mfcc_coff, axis=0)
    std_mfcc = np.std(mfcc_coff, axis=0)

    # Plot error bars with a connecting line for each coefficient index
    plt.errorbar(range(mean_mfcc.shape[0]), mean_mfcc, yerr=std_mfcc, fmt='o',color="red", ecolor='red', capsize=5)
    plt.plot(range(mean_mfcc.shape[0]), mean_mfcc, linestyle='-', color='black')  # Line to connect points
    plt.xlabel("Coefficient Index")
    plt.ylabel("MFCC Coefficient Value")
    plt.title(f"{rl}")
    plt.show()

