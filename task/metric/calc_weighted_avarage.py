import pandas as pd

# Assuming the CSV file path
csv_file_path = "./weigth_average_score.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Normalize the scores to a 0-1 scale
max_score = df['score'].max()  # Find the maximum score in the dataset
df['normalized_score'] = df['score'] / max_score  # Normalize scores

# Now, you can calculate the weighted average of the normalized scores.
# If each song has the same weight, this simplifies to the mean of the normalized scores.
weighted_average_normalized_score = df['normalized_score'].mean()

print(f"Weighted Average Normalized Completeness Score: {weighted_average_normalized_score}")
