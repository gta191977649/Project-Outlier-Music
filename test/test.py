import madmom, scipy.stats, numpy as np
from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label
from madmom.features.chords import DeepChromaChordRecognitionProcessor, CRFChordRecognitionProcessor, \
    CNNChordFeatureProcessor

# Load the audio file
audio_file_path = "F:\\dataset\\custom\\Toto - Africa (Official HD Video).mp3"
proc = CNNKeyRecognitionProcessor()(audio_file_path)
detected = key_prediction_to_label(proc)
print(detected)
