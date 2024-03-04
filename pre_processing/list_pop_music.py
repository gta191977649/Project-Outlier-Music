import pandas as pd
import os
# Path to your CSV file
CSV_LOC = "/Users/nurupo/Desktop/dev/Project-Outlier-Music/dataset/music4all/dataset.csv"
# Define the output file path in the same directory but with a different filename
output_csv_path = os.path.join(os.path.dirname(CSV_LOC), "pop_songs.csv")

# Load the dataset
df = pd.read_csv(CSV_LOC)

# Assuming the column containing tags is named 'tags'
# Filter rows where the 'tags' column contains 'pop'
pop_songs = df[df['tags'].str.contains("pop", case=False, na=False)]

# Save the filtered DataFrame to the new CSV file
pop_songs.to_csv(output_csv_path, index=False)

print(f"Songs with 'pop' tags have been saved to {output_csv_path}")