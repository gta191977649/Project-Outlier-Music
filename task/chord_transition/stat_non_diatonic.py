import os
import pandas as pd
from model.song import Song
import matplotlib.pyplot as plt
import seaborn as sns

# Configurable year interval for grouping
YEAR_INTERVAL = 5

def countChordTransition(roman_labels, matches=1):
    diatonic_to_nondiatonic = 0
    nondiatonic_to_diatonic = 0
    nondiatonic_to_nondiatonic = 0

    for i in range(len(roman_labels) - 1):
        current_chord = roman_labels[i]
        next_chord = roman_labels[i + 1]

        if current_chord != "?" and next_chord == "?":
            diatonic_to_nondiatonic += 1
        elif current_chord == "?" and next_chord != "?":
            nondiatonic_to_diatonic += 1
        elif current_chord == "?" and next_chord == "?":
            nondiatonic_to_nondiatonic += 1

    return {
        "diatonic_to_nondiatonic": diatonic_to_nondiatonic * matches,
        "nondiatonic_to_diatonic": nondiatonic_to_diatonic * matches,
        "nondiatonic_to_nondiatonic": nondiatonic_to_nondiatonic * matches
    }

def group_years(year):
    # Adjust the year to fit into the corresponding group
    return (year // YEAR_INTERVAL) * YEAR_INTERVAL

if __name__ == '__main__':
    PATH = "/Users/nurupo/Desktop/dev/music4all/"
    audio_folder = "audios"
    path_csv = os.path.join(PATH, "stratified_songs_pop.csv")

    song_collections = []

    if os.path.exists(path_csv):
        csv = pd.read_csv(path_csv)
        for _, item in csv.iterrows():
            id = item['id']
            h5_path = os.path.join(PATH, audio_folder, id + ".h5")
            song = Song.from_h5(h5_path)
            song_collections.append(song)

    data_by_year_group = {}
    artist_non_diatonic_songs_by_year_group = {}

    for song in song_collections:
        year = int(song.release)
        year_group = group_years(year)

        if year_group not in data_by_year_group:
            data_by_year_group[year_group] = {
                'total_songs': 0,
                'total_detected_chords': 0,
                'total_nondiatonic': 0,
                'diatonic_to_nondiatonic': 0,
                'nondiatonic_to_diatonic': 0,
                'nondiatonic_to_nondiatonic': 0,
                'notable_outlier_artist': []  # [(artist_name, score)]
            }

        total_chords = diatonic_to_nondiatonic = nondiatonic_to_diatonic = nondiatonic_to_nondiatonic = 0

        if len(song.chord_pattern) > 0:
            for ptn in song.chord_pattern:
                total_chords += len(ptn["pattern"])
                matches = int(ptn["matches"])
                result = countChordTransition(ptn["roman_label"], matches)
                diatonic_to_nondiatonic += result["diatonic_to_nondiatonic"]
                nondiatonic_to_diatonic += result["nondiatonic_to_diatonic"]
                nondiatonic_to_nondiatonic += result["nondiatonic_to_nondiatonic"]

        total_nondiatonic = diatonic_to_nondiatonic + nondiatonic_to_diatonic + nondiatonic_to_nondiatonic
        data_by_year_group[year_group]['total_detected_chords'] += total_chords
        data_by_year_group[year_group]['diatonic_to_nondiatonic'] += diatonic_to_nondiatonic
        data_by_year_group[year_group]['nondiatonic_to_diatonic'] += nondiatonic_to_diatonic
        data_by_year_group[year_group]['nondiatonic_to_nondiatonic'] += nondiatonic_to_nondiatonic
        data_by_year_group[year_group]['total_nondiatonic'] += total_nondiatonic
        data_by_year_group[year_group]['total_songs'] += 1

        # Artist count logic
        if total_nondiatonic > 0:
            if year_group not in artist_non_diatonic_songs_by_year_group:
                artist_non_diatonic_songs_by_year_group[year_group] = {}

            artist_non_diatonic_songs_by_year_group[year_group].setdefault(song.artist, 0)
            artist_non_diatonic_songs_by_year_group[year_group][song.artist] += total_nondiatonic

    # Normalize the stats and find notable artists
    for year_group in data_by_year_group:
        if data_by_year_group[year_group]['total_nondiatonic'] > 0:
            data_by_year_group[year_group]["diatonic_to_nondiatonic"] /= data_by_year_group[year_group]['total_nondiatonic']
            data_by_year_group[year_group]["nondiatonic_to_diatonic"] /= data_by_year_group[year_group]['total_nondiatonic']
            data_by_year_group[year_group]["nondiatonic_to_nondiatonic"] /= data_by_year_group[year_group]['total_nondiatonic']

        artists = artist_non_diatonic_songs_by_year_group.get(year_group, {})
        top_artists = sorted(artists.items(), key=lambda x: x[1], reverse=True)[:3]
        data_by_year_group[year_group]['notable_outlier_artist'] = top_artists

    df = pd.DataFrame.from_dict(data_by_year_group, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Year Group'}, inplace=True)
    df.to_csv("/Users/nurupo/Desktop/dev/music4all/stat_chord_transitions_grouped.csv", index=False)
    print(df)
