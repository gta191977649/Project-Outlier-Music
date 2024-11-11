import pandas as pd

# Load the CSV file
file_path = '/Users/nurupo/Desktop/dev/msd/song_v2.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Group by artist and summarize statistics
artist_summary = df.groupby('artist').agg(
    num_songs=('id', 'count'),
    avg_loudness=('loudness', 'mean'),
    avg_tempo=('tempo', 'mean'),
    min_loudness=('loudness', 'min'),
    max_loudness=('loudness', 'max'),
    min_tempo=('tempo', 'min'),
    max_tempo=('tempo', 'max')
).reset_index().sort_values(by='num_songs', ascending=False)


# Print the summary
print(artist_summary)

artist_summary.to_csv('/Users/nurupo/Desktop/dev/msd/song_stat.csv', index=False)
