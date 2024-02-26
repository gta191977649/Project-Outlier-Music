import pandas as pd
import os

# Load the dataset
CSV_LOC = "/Users/nurupo/Desktop/dev/music4all/dataset.csv"

df = pd.read_csv(CSV_LOC)

# Filter the dataset for songs released between 1980 and 2019 and whose genre includes 'pop'
df_filtered = df[(df['release'] >= 1980) & (df['release'] <= 2019) & df['tags'].str.contains('pop', case=False, na=False)]

# Group by artist and count the number of unique release years
artist_years = df_filtered.groupby('artist')['release'].nunique()

# Get the top 10 artists with the longest span of song releases
top_artists_longest_span = artist_years.sort_values(ascending=False).head(10)
print(top_artists_longest_span)
# The variables 'longest_span_artist' and 'longest_span_years' now hold the artist with the longest span and the number of years
