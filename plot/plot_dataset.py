import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'BIZ UDGothic'
plt.rcParams['font.size'] = 12
# Load the data
df = pd.read_csv('/Users/nurupo/Desktop/dev/music4all/pop_songs.csv')

# Group by 'release' and count the number of songs per year
n_songs_year = df.groupby('release').size()

# Count the number of unique artists per year
n_artists_year = df.groupby('release')['artist'].nunique()
n_unique_artists = df['song'].nunique()
print(n_unique_artists)
# Combine into a single DataFrame
combined_data = pd.concat([n_songs_year, n_artists_year], axis=1)
combined_data.columns = ['Number of Songs', 'Number of Artists']

# Configurable year range
start_year = 1980
end_year = 2019

# Filter the DataFrame for the desired year range
filtered_data = combined_data.loc[start_year:end_year]

# Plotting
plt.figure(figsize=(10, 4))

# Adjust the index for plotting since we've filtered the data
index = range(len(filtered_data))

bar_width = 0.35
plt.bar(index, filtered_data['Number of Songs'], bar_width, label='Songs', color='#F5F5F5',edgecolor='black', linewidth=1)
plt.bar([p + bar_width for p in index], filtered_data['Number of Artists'], bar_width, label='Artists', color='#F8CECC',edgecolor='black', linewidth=1)

plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Number of Pop Songs and Unique Artists per Year')

# Adjust the ticks to show the filtered years
plt.xticks([p + bar_width / 2 for p in index], filtered_data.index, rotation=90)

# Automatically adjusts the x-axis limits based on the filtered data
plt.xlim(-1, len(filtered_data) + 1)

plt.legend()
plt.tight_layout()
plt.show()
