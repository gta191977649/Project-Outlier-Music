import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV
path_csv = '/Users/nurupo/Desktop/dev/Project-Outlier-Music/results/transition_outliers.csv'
df = pd.read_csv(path_csv)

# Initialize a dictionary to count occurrences of each category
category_counts = {}

# Iterate over the 'Category' column
for categories in df['Category']:
    # Check if categories is not NaN (float)
    if isinstance(categories, str):
        # Split the categories by ',' and strip whitespace
        split_categories = [cat.strip() for cat in categories.split(',')]
        for category in split_categories:
            # Count each category
            if category not in category_counts:
                category_counts[category] = 1
            else:
                category_counts[category] += 1

# Convert the counts to a DataFrame for easier plotting
categories_df = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count']).sort_values(by='Count', ascending=False)

# Generate a color palette with a distinct color for each category
num_categories = len(categories_df)
colors = plt.cm.tab10(np.linspace(0, 1, num_categories))

# Plot
plt.figure(figsize=(6, 5))
plt.bar(categories_df['Category'], categories_df['Count'], color=colors, edgecolor="black")
plt.xlabel('Category')
plt.ylabel('Number of Songs')
plt.title('Outlier Category')
plt.xticks(rotation=90, ha="right")
plt.tight_layout()
plt.style.use("classic")
plt.show()
