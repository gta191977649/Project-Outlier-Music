from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
import numpy as np

# Function to perform downbeat tracking with a given beats_per_bar guess
# def perform_downbeat_tracking(beats_per_bar):
#     downbeat_processor = RNNDownBeatProcessor()
#     downbeat_tracking_processor = DBNDownBeatTrackingProcessor(fps=100,beats_per_bar=beats_per_bar)
#     downbeats = downbeat_tracking_processor(downbeat_processor(audio_file))
#     return downbeats

def evaluate_downbeat_consistency(downbeats):
    # Calculate intervals between consecutive downbeats and their standard deviation
    intervals = np.diff(downbeats)
    std_deviation = np.std(intervals)
    return std_deviation
def getTempoMarkingEncode(tempo):
    if tempo > 200:
        return 7
    elif 168 <= tempo <= 200:
        return 6
    elif 120 <= tempo < 168:
        return 5
    elif 108 <= tempo < 120:
        return 4
    elif 76 <= tempo < 108:
        return 3
    elif 66 <= tempo < 76:
        return 2
    elif 40 <= tempo < 66:
        return 1
    elif 20 <= tempo < 40:
        return 0
    else:
        return "Tempo out of range"
def getTempoMarking(tempo):
    if tempo > 200:
        return "Prestissimo"
    elif 168 <= tempo <= 200:
        return "Presto"
    elif 120 <= tempo < 168:
        return "Allegro"
    elif 108 <= tempo < 120:
        return "Moderato"
    elif 76 <= tempo < 108:
        return "Andante"
    elif 66 <= tempo < 76:
        return "Adagio"
    elif 40 <= tempo < 66:
        return "Lento/Largo"
    elif 20 <= tempo < 40:
        return "Grave"
    else:
        return "Tempo out of range"