import os
import feature.msd_getter as h5
import matplotlib.pyplot as plt
import speechpy as speechpy
import numpy as np
from matplotlib.cm import get_cmap

def plot_data(data):
    num_releases = len(data)
    cmap = get_cmap('nipy_spectral', num_releases)  # Generates a non-repeating color map

    plt.figure(figsize=(12, 6))
    for idx, (release, metrics) in enumerate(data.items()):
        color = cmap(idx)
        plt.scatter(metrics['mfcc_mean_first'], metrics['mfcc_mean'], color=color, label=release)

    plt.xlabel('Tempo')
    plt.ylabel('MFCC')
    plt.title('Tempo vs Loudness by Release')
    plt.legend(title="Releases", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
            releases[release].append(filename)

    return releases


def collect_mfcc(directory, release_group):
    mfcc_coeffs = []
    # Iterate over filenames in the specific release group
    for filename in release_group:
        full_path = os.path.join(directory, filename)
        song = h5.open_h5_file_read(full_path)
        mfcc = h5.get_segments_timbre(song).T  # Assuming it returns a 2D array where columns are MFCCs
        # Exclude the first MFCC coefficient (index 0), which is the DC component

        # Calculate the mean of each MFCC coefficient across time
        mfcc_means = np.mean(mfcc, axis=1)
        mfcc_coeffs.append(mfcc_means)

    return np.array(mfcc_coeffs)


def plot_mfcc_by_index(mfcc_coeffs):
    plt.figure(figsize=(10, 8))
    # Transpose to make each row an index across all songs
    transposed_mfcc = mfcc_coeffs.T

    # Plot each index as a separate line
    for idx, coeffs in enumerate(transposed_mfcc):
        plt.plot(coeffs, label=f'MFCC Index {idx}')

    plt.title('MFCC Coefficients Grouped by Index Across Songs')
    plt.xlabel('Song Number')
    plt.ylabel('Mean MFCC Value')
    plt.legend(title="MFCC Indices", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    path = "/Users/nurupo/Desktop/dev/msd/blue_oyster"
    grouped_releases = group_files_by_release(path)
    for release, filenames in grouped_releases.items():
        print(release,len(filenames))
    # printout all release and file names
    # data = {}
    # for release, filenames in grouped_releases.items():
    #     for filename in filenames:
    #         full_path = os.path.join(path, filename)
    #         song = h5.open_h5_file_read(full_path)
    #         release = h5.get_release(song).decode("utf-8")
    #         tempo = h5.get_tempo(song)
    #         loudness = h5.get_loudness(song)
    #         mfcc = h5.get_segments_timbre(song).T
    #         # normalize mfcc
    #         #mfcc = speechpy.processing.cmvn(mfcc, False)
    #         mfcc_mean_first = np.mean(mfcc[0])
    #         mfcc_mean_second = np.mean(mfcc[1])
    #         mfcc_mean_third = np.mean(mfcc[2])
    #         mfcc_mean = np.mean(mfcc)
    #         if release not in data:
    #             data[release] = {
    #                 'tempo':[],
    #                  'loudness': [],
    #                  'mfcc_mean': [],
    #                  'mfcc_mean_first': [],
    #                  'mfcc_mean_second':[],
    #                  'mfcc_mean_third':[]
    #              }
    #         data[release]['tempo'].append(tempo)
    #         data[release]['loudness'].append(loudness)
    #         data[release]['mfcc_mean'].append(mfcc_mean)
    #         data[release]['mfcc_mean_first'].append(mfcc_mean_first)
    #         data[release]['mfcc_mean_second'].append(mfcc_mean_second)
    #         data[release]['mfcc_mean_third'].append(mfcc_mean_third)
    #
    # plot_data(data)
    #rl = "Secret Treaties"
    rl = "Extraterrestrial Live"
    mfcc_coffs = []
    for filename in grouped_releases[rl]:
        song = h5.open_h5_file_read(os.path.join(path, filename))
        title = h5.get_title(song)
        mfcc = h5.get_segments_timbre(song)
        # Some magic normalization used from speaker identification
        mfcc = speechpy.processing.cmvn(mfcc.T,True)
        mfcc = np.array(mfcc).T
        # normalize mfcc for all 12 coffs
        mfcc_norm = np.mean(mfcc,axis=0)

        mfcc_coffs.append(mfcc_norm)

    mfcc_coffs = np.array(mfcc_coffs).T# tranpose this shit for plot
    plt.plot(mfcc_coffs)
    plt.title(rl)
    plt.xlim(0,11)
    plt.show()
        #print(title,tempo)


