import pandas as pd
import os
# Load the dataset
csv_info_path = "/Users/nurupo/Desktop/dev/music4all/pop_songs.csv"
df = pd.read_csv(csv_info_path)
output_csv_path = os.path.join(os.path.dirname(csv_info_path), "stratified_songs_pop_2.csv")

# Convert 'release' to integer if it's not already
df['release'] = df['release'].astype(int)

# Filter the dataset to include only the years from 1980 onwards
df_filtered = df[df['release'] >= 1980]

# Find the smallest number of songs from 1980 onwards
min_songs_per_year = df_filtered['release'].value_counts().min()

# Perform stratified sampling
stratified_sample = df_filtered.groupby('release', group_keys=False).apply(
    lambda x: x.sample(min(len(x), min_songs_per_year), random_state=None)
)

# Save the stratified sample to a new CSV file
stratified_sample.to_csv(output_csv_path, index=False)

print(f"Stratified sample from 1980 onwards saved to {output_csv_path}")
