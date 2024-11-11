import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV
csv_tempo = pd.read_csv("/Users/nurupo/Desktop/dev/Project-Outlier-Music/task/chap_7/in-album_outlier_loudness.csv")

# Count the relationships between origin_context and fit_context
relationship_counts = csv_tempo.groupby(['origin_context', 'fit_context']).size().unstack(fill_value=0)
#relationship_counts = relationship_counts[:100]
# Truncate labels to a maximum of 6 characters
short_index_labels = [label[:6] for label in relationship_counts.index]
short_column_labels = [label[:6] for label in relationship_counts.columns]

# Plot the heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(
    relationship_counts,
    annot=False,
    cbar=False,
    xticklabels=False,  # Disable automatic x-tick labels
    yticklabels=False   # Disable automatic y-tick labels
)

# Set x-ticks and y-ticks at specific intervals
step = 20  # Show labels every 5th tick
x_ticks = range(0, len(short_column_labels), step)
y_ticks = range(0, len(short_index_labels), step)

plt.xticks(
    ticks=x_ticks,
    labels=[short_column_labels[i] for i in x_ticks],
    rotation=45,  # Rotate x-tick labels for readability
    ha='right'
)
plt.yticks(
    ticks=y_ticks,
    labels=[short_index_labels[i] for i in y_ticks]
)

plt.title("Cross-Artist Relationships Matrix - (Feature: Loudness)")
plt.xlabel("Fit Context (Linked Artist)")
plt.ylabel("Origin Context (Source Artist)")
plt.tight_layout()
plt.show()
