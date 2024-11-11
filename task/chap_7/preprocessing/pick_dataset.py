import pandas as pd
import random

def load_existing_data(filepath):
    # Load the existing dataset from a CSV file
    return pd.read_csv(filepath)

def create_subset_dataset(df, artists, num_songs_per_artist):
    # Ensure we work only with artists present in the dataset
    available_artists = df['artist'].unique()
    filtered_artists = [artist for artist in artists if artist in available_artists]

    # Generate the dataset
    data = []
    for artist in filtered_artists:
        artist_songs = df[df['artist'] == artist]
        if len(artist_songs) >= num_songs_per_artist:
            sampled_songs = artist_songs.sample(n=num_songs_per_artist)
        else:
            # If there aren't enough songs, repeat songs to fulfill the requirement
            sampled_songs = artist_songs.sample(n=num_songs_per_artist, replace=True)
        data.append(sampled_songs)

    # Concatenate all sampled data into a single DataFrame
    result_df = pd.concat(data, ignore_index=True)
    return result_df

# Load existing data
existing_data = load_existing_data("/Users/nurupo/Desktop/dev/msd/song_v2.csv")  # Provide your actual input file path here

# Inputs
artists = ["Blue Oyster Cult", "Lady Bug Transistor", "DeBarge"]  # Replace with your list of artist names
num_songs_per_artist = 30  # Specify the number of songs per artist

# Generate subset dataset
subset_dataset = create_subset_dataset(existing_data, artists, num_songs_per_artist)

# Save to CSV
subset_dataset.to_csv('/Users/nurupo/Desktop/dev/msd/song_picked.csv', index=False)
print("Subset dataset saved as 'subset_songs_dataset.csv'")
