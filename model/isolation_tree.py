import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

from feature.chord import *

class ChordProgressionIsolationForest:
    def __init__(self, chord_progressions):
        self.chord_progressions = chord_progressions
        self.vectorized_progressions = self._vectorize_progressions()
        self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        print(self.vectorized_progressions)

    def _vectorize_progressions(self):
        return np.array([
            extractChordNumeralValues(progression)
            for progression in self.chord_progressions
        ])

    def detect_outliers(self):
        self.outlier_detector.fit(self.vectorized_progressions)
        return self.outlier_detector.predict(self.vectorized_progressions)

    def compute_anomaly_scores(self):
        return -self.outlier_detector.score_samples(self.vectorized_progressions)

    def visualize_anomaly_scores(self):
        anomaly_scores = self.compute_anomaly_scores()

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(self.chord_progressions))

        ax.bar(x, anomaly_scores, color='green', alpha=0.7)
        ax.set_ylabel('Anomaly Score')
        ax.set_xlabel('Chord Progressions')
        ax.set_title('Anomaly Scores for Chord Progressions')
        ax.set_xticks(x)
        #ax.set_xticklabels(['-'.join(prog) for prog in self.chord_progressions], rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

    def print_outliers(self):
        outliers = self.detect_outliers()
        anomaly_scores = self.compute_anomaly_scores()

        print("Outliers detected:")
        for i, (is_outlier, score) in enumerate(zip(outliers, anomaly_scores)):
            if is_outlier == -1:  # -1 indicates an outlier in Isolation Forest
                print(f"Progression {i}: {self.chord_progressions[i]}")
                print(f"Anomaly Score: {score}")
                print("--------------------")

