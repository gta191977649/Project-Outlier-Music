from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
import numpy as np

audio_file = '/Users/nurupo/Desktop/dev/music4all/custom/MV君はメロディー Short ver.  AKB48[公式].mp3'

# Perform beat tracking
beat_processor = RNNBeatProcessor()
beats = beat_processor(audio_file)

# Potential initial guesses for beats per bar
initial_beats_per_bar_options = [3, 4, 6]

# Function to perform downbeat tracking with a given beats_per_bar guess
def perform_downbeat_tracking(beats_per_bar):
    downbeat_processor = RNNDownBeatProcessor()
    downbeat_tracking_processor = DBNDownBeatTrackingProcessor(fps=100,beats_per_bar=beats_per_bar)
    downbeats = downbeat_tracking_processor(downbeat_processor(audio_file))
    return downbeats

# Dictionary to hold downbeat tracking results for each beats_per_bar guess
downbeat_results = {}

# Try downbeat tracking with each initial guess and store the results
for option in initial_beats_per_bar_options:
    downbeats = perform_downbeat_tracking(option)
    downbeat_results[option] = downbeats

# Placeholder for logic to analyze downbeat results and select the best guess
# For demonstration, this part needs to be filled in with specific logic to evaluate the best fit

# For now, let's assume we simply select the option that produces the most consistent downbeat intervals
# This is a simplification and might not accurately reflect the best method for all types of music

def evaluate_downbeat_consistency(downbeats):
    # Calculate intervals between consecutive downbeats and their standard deviation
    intervals = np.diff(downbeats)
    std_deviation = np.std(intervals)
    return std_deviation

best_option = min(initial_beats_per_bar_options, key=lambda x: evaluate_downbeat_consistency(downbeat_results[x]))

print(f"Best initial guess for beats_per_bar: {best_option}")
