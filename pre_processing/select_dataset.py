import pandas as pd

# Load the dataset
CSV_LOC =  "/Users/nurupo/Desktop/dev/music4all/dataset.csv"
df = pd.read_csv(CSV_LOC)


# Function to filter songs by artist and save to CSV
def filter_songs_by_artist(artist_name, output_csv_path):
    # Filter the DataFrame for the given artist
    df_artist_songs = df[df['artist'].str.lower() == artist_name.lower()]

    # Save the filtered DataFrame to a new CSV file
    df_artist_songs.to_csv(output_csv_path, index=False)
    print(f"Songs by artist '{artist_name}' have been saved to {output_csv_path}")


# Example usage:
artist_to_filter = 'Depeche Mode'  # Replace with the artist's name you want to filter by
output_csv_path =  "/Users/nurupo/Desktop/dev/music4all/Depeche_Mode.csv"

# Call the function with the artist name and output path
filter_songs_by_artist(artist_to_filter, output_csv_path)
