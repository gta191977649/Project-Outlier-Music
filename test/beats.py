from madmom.features import CNNChordFeatureProcessor, CRFChordRecognitionProcessor, RNNDownBeatProcessor, DBNDownBeatTrackingProcessor

audio_file_name = "/Users/nurupo/Desktop/dev/music4all/custom/a-ha - Take On Me (Official Video) [Remastered in 4K].mp3"

chord_processor = CNNChordFeatureProcessor()
chord_decoder = CRFChordRecognitionProcessor()
chords = chord_decoder(chord_processor(audio_file_name))

beat_processor = RNNDownBeatProcessor()
beat_decoder = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)
beats = beat_decoder(beat_processor(audio_file_name))

beat_idx = 0
for chord_idx in range(len(chords)):
  chord_time, _, chord = chords[chord_idx]

  if chord == 'N':
    continue

  min_diff = float("inf")
  for i in range(beat_idx, len(beats)):
    beat_time, beat = beats[i]
    if int(beat) != 1:
      continue
    if abs(chord_time - beat_time) < min_diff:
      min_diff = abs(chord_time - beat_time)
      beat_idx = i
    else:
      break

  beat_time, bt = beats[beat_idx]
  print(chord_time, chord, bt, chord_time - beat_time)