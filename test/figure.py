import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sample data
scores_3_beats = np.random.normal(70, 10, 100)
scores_4_beats = np.random.normal(75, 8, 100)

plt.figure(figsize=(10, 6))
sns.boxplot(data=[scores_3_beats, scores_4_beats], palette="Set2")
plt.xticks([0, 1], ['3 Beats per Bar', '4 Beats per Bar'])
plt.ylabel('Completeness Score')
plt.title('Completeness Score Distribution')
plt.show()