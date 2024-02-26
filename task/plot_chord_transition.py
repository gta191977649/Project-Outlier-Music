import os
import pandas as pd
from model.song import Song
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    PATH = "/Users/nurupo/Desktop/dev/music4all/"
    audio_folder = "audios"
    path_csv = os.path.join(PATH, "stratified_songs_pop.csv")

    collection = []
    if os.path.exists(path_csv):
        csv = pd.read_csv(path_csv)
        for _, item in csv.iterrows():
            id = item['id']
            h5_path = os.path.join(PATH, audio_folder, id + ".h5")
            song = Song.from_h5(h5_path)
            collection.append(song)

    non_diatonic_counts_by_year = {}
    total_transitions_by_year = {}

    for song in collection:
        year = int(song.release)
        if not song.mode == "major": continue
        for pattern in song.chord_pattern:
            # Skip invalid chord data
            if "N" in pattern.get("pattern"):
                print("SKIP INVALID CHORD DATA")
                continue

            non_diatonic_count = pattern.get("non_diatonic_chords_count", 0)
            total_transitions = len(pattern.get("pattern", ""))

            # Update counts
            non_diatonic_counts_by_year[year] = non_diatonic_counts_by_year.get(year, 0) + non_diatonic_count
            total_transitions_by_year[year] = total_transitions_by_year.get(year, 0) + total_transitions

    # Calculate the normalized ratio of non_diatonic_counts to total_transitions for each year
    normalized_ratio_by_year = {year: non_diatonic_counts_by_year[year] / total_transitions_by_year[year]
    if total_transitions_by_year[year] > 0 else 0
                                for year in non_diatonic_counts_by_year}

    # Convert the dictionary to a DataFrame for easier plotting
    df_normalized_ratio = pd.DataFrame(list(normalized_ratio_by_year.items()), columns=['Year', 'Normalized_Ratio'])

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Year', y='Normalized_Ratio', data=df_normalized_ratio, palette='viridis')
    plt.title('Normalized Ratio of Non-Diatonic Transitions by Year', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Normalized Ratio (0-1)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()
