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

def to_harte_label(chord):
    chord = chord.replace('maj', ':maj').replace('min', ':min').replace('dim', ':dim').replace('aug', ':aug')
    return chord


def transpose_signal(X):
    """
    Normalize the signal to create a 2D feature matrix.

    Parameters:
    signal (list or np.array): 1D array of tonal pitch distance pattern values.

    Returns:
    np.array: 2D feature matrix where rows represent time frames and columns represent tonal distances.
    """
    num_time_frames = len(X)
    num_tonal_distances = 14  # Since tonal distances range from 0 to 13
    X_norm = np.zeros((num_time_frames, num_tonal_distances))

    for i, val in enumerate(X):
        val = int(val)
        if 0 <= val < num_tonal_distances:
            X_norm[i, val] = 1

    return X_norm.T

def compute_kernel_checkerboard_gaussian(L, var=1, normalize=True):
    """Compute Guassian-like checkerboard kernel [FMP, Section 4.4.1].
    See also: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        L (int): Parameter specifying the kernel size M=2*L+1
        var (float): Variance parameter determing the tapering (epsilon) (Default value = 1.0)
        normalize (bool): Normalize kernel (Default value = True)

    Returns:
        kernel (np.ndarray): Kernel matrix of size M x M
    """
    taper = np.sqrt(1/2) / (L * var)
    axis = np.arange(-L, L+1)
    gaussian1D = np.exp(-taper**2 * (axis**2))
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    kernel_box = np.outer(np.sign(axis), np.sign(axis))
    kernel = kernel_box * gaussian2D
    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))
    return kernel
def compute_novelty_ssm(S, kernel=None, L=10, var=0.5, exclude=False):
    """Compute novelty function from SSM [FMP, Section 4.4.1]

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        S (np.ndarray): SSM
        kernel (np.ndarray): Checkerboard kernel (if kernel==None, it will be computed) (Default value = None)
        L (int): Parameter specifying the kernel size M=2*L+1 (Default value = 10)
        var (float): Variance parameter determing the tapering (epsilon) (Default value = 0.5)
        exclude (bool): Sets the first L and last L values of novelty function to zero (Default value = False)

    Returns:
        nov (np.ndarray): Novelty function
    """
    if kernel is None:
        kernel = compute_kernel_checkerboard_gaussian(L=L, var=var)
    N = S.shape[0]
    M = 2*L + 1
    nov = np.zeros(N)
    # np.pad does not work with numba/jit
    S_padded = np.pad(S, L, mode='constant')

    for n in range(N):
        # Does not work with numba/jit
        nov[n] = np.sum(S_padded[n:n+M, n:n+M] * kernel)
    if exclude:
        right = np.min([L, N])
        left = np.max([0, N-L])
        nov[0:right] = 0
        nov[left:N] = 0

    return nov




def plot_function_peak_positions(f_novel, f_picks):
    """
    Plot the novelty function with peak positions highlighted.

    Parameters:
    f_novel (np.array): Novelty function.
    f_picks (np.array): Indices of peak positions.
    """
    plt.figure(figsize=(15, 3))
    plt.plot(f_novel,color='b', label='Novelty function')
    plt.scatter(f_picks, f_novel[f_picks], color='r', zorder=3)
    for peak in f_picks:
        plt.axvline(x=peak, color='red', linestyle='dotted', linewidth=1)
    plt.title('Novelty Function with Peak Positions')
    plt.xlabel('Time (frames)')
    plt.ylabel('Novelty')
    plt.xlim([0, len(f_novel)])
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_ssm_with_novelty(X_trans, S, f_novel):
    fig = plt.figure(figsize=(10, 15))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 6, 1])

    # Plot the feature sequence
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(X_trans, aspect='auto', cmap='gray_r', origin='lower')
    ax0.set_title('Feature Sequence')
    ax0.set_xlabel('Time (frames)')
    ax0.set_ylabel('Feature Index')

    # Plot the SSM
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(S, aspect='auto', cmap='gray_r', origin='lower')
    ax1.set_title('Self-Similarity Matrix (SSM)')
    ax1.set_xlabel('Time (frames)')
    ax1.set_ylabel('Time (frames)')
    ax1.set_aspect('equal')

    # Plot the novelty function
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(f_novel,color='b')
    ax2.set_title('Novelty Function')
    ax2.set_xlabel('Time (frames)')
    ax2.set_ylabel('Novelty')
    ax2.set_xlim([0, len(f_novel)])

    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    TARGET_MODE = "major"
    KEY = "B:maj"

    PATH = "/Users/nurupo/Desktop/dev/music4all/akb48/君はメロディー [9Spu8vH0eUs].h5"

    song = Song.from_h5(PATH)
    #song = Song(1, "/Users/nurupo/Desktop/dev/music4all/custom/MV君はメロディー Short ver.  AKB48[公式].h5", "Living On The Ceiling","Blancmange")

    chords = []
    for item in song.chord:
        _,_,chord = item
        chords.append(chord)

    signal = extractTontalPitchDistancePattern(chords, mode="profile",key=KEY)
    #signal = extractChromaticPattern(chords)
    N = len(chords)
    print(f"N is {N}")
    signal = signal[:N]
    chords = chords[:N]

    X_trans = transpose_signal(signal)


    #Compute SSM
    #S = np.dot(X_trans.T, X_trans)
    N = len(chords)
    S = np.zeros((N, N))

    for n in range(0,S.shape[0]):
        for m in range(0,S.shape[1]):
            S[n,m] = 1 - (computeTPSD(chords[n],chords[m],key=KEY) / 13)

    #Compute Novelty function
    f_novel = compute_novelty_ssm(S,L=20, exclude=True)
    #f_picks = scipy.signal.find_peaks(f_novel,distance=30)[0]
    f_novel_sm = filters.gaussian_filter1d(f_novel, sigma=2)

    f_picks = librosa.util.peak_pick(f_novel_sm, pre_max=5, post_max=5, pre_avg=5, post_avg=5,delta=0.01, wait=10)

    # Plot SSM with novelty func
    plot_ssm_with_novelty(X_trans, S, f_novel)

    plot_function_peak_positions(f_novel_sm,f_picks)






    plt.figure(figsize=(15, 3))
    plt.step(range(len(signal)), signal, where='mid', color='b', linewidth=1.5)
    plt.xlim([0, len(signal)])
    # plt.title(format_chord_progression(chord_progression["pattern"]))
    plt.xlabel('Chord Index')
    plt.ylabel('Tonal Pitch Distance (TPSD)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    #for chord_progression in song.chord_pattern:
