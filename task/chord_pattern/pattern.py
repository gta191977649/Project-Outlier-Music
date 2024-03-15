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

if __name__ == '__main__':
    TARGET_MODE = "major"
    PATH = "/Users/nurupo/Desktop/dev/music4all/akb48/"
    # loop all folder
    song_collections = []
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith(".h5"):
                filepath = os.path.join(root, file)
                filename = os.path.splitext(file)[0]
                song = Song.from_h5(filepath)
                if song.mode == TARGET_MODE:
                    song_collections.append(song)

    # Prepare pattern dataset ...
    X_train = []
    X_Label = []
    X_Chord_Label = []
    for song in song_collections:
        for pat in song.chord_pattern:
            progression = pat["pattern"]
            roman = pat["roman_label"]
            thetas = []
            for chord in progression:
                # Do Chord Transpose Due to Patterns are on original key
                #chord = extract.transposeChordLabel(chord,song.transpose_amount)
                angle = extract.getChordVectorsAngleFromChord(chord)
                thetas.append(angle)
            #print(thetas)
            X_train.append(thetas)
            X_Label.append(roman)
            X_Chord_Label.append(progression)
    # clustering
    pattern_cluster ={}
    for i in range(len(X_train)):
        reference_signal = X_train[i]
        pattern_cluster[i] = 0
        for j in range(len(X_train)):
            if not i == j:
                search_signal = X_train[j]
                cost = dtw(reference_signal, search_signal)
                # normalize cost
                path_length = len(reference_signal) + len(search_signal)
                cost = cost / path_length
                if cost < 5:
                    pattern_cluster[i] +=1

    # for key , value in pattern_cluster.items():
    #     print(X_Label[key], value)



    # Step 1: Count the occurrence frequency of each unique chord progression
    # occurrence_frequency = pattern_cluster.values()
    #
    # # Step 2: Calculate the total number of progressions in the dataset
    # total_progressions = len(X_Chord_Label)
    # print(f"Total progressions: {total_progressions}\n")
    #
    # # Step 3: Calculate the relative frequency of each progression
    # relative_frequency = {pattern: freq / total_progressions for pattern, freq in pattern_cluster.items()}
    #
    # # Step 4: Identify common chord progressions
    # commonality_threshold = 0.5
    # common_progressions = {pattern: freq for pattern, freq in relative_frequency.items() if
    #                        freq >= commonality_threshold}
    #
    # print("Common chord progressions (relative frequency ≥ 0.01):")
    # for i, (pattern, freq) in enumerate(common_progressions.items(), 1):
    #     print(f"{i}. {X_Label[pattern]} ({freq:.4f})")
    #
    # print()
    #
    # # Step 5: Identify outlier chord progressions
    # outlier_progressions = {pattern: freq for pattern, freq in relative_frequency.items() if
    #                         freq < commonality_threshold}
    #
    # print("Outlier chord progressions (relative frequency < 0.01):")
    # print(
    #     "- All other progressions in the dataset that are not listed as common progressions above are considered outliers.")
    #
    # print("\nSummary:")
    # print(f"Out of the {total_progressions} total chord progressions in the dataset:")
    # print(
    #     f"- There are {len(common_progressions)} common chord progressions that appear with a relative frequency of at least {commonality_threshold}.")
    # print(
    #     f"- All other progressions ({len(outlier_progressions)}) that appear with a relative frequency below {commonality_threshold} are considered outliers.")
