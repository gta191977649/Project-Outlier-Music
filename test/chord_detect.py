import essentia.streaming as ess
import essentia
from essentia.standard import ChordsDetection

audio_file = '/Users/nurupo/Desktop/music/c_major.mp3'

# Initialize algorithms we will use.
loader = ess.MonoLoader(filename=audio_file)
framecutter = ess.FrameCutter(frameSize=4096, hopSize=2048, silentFrames='noise')
windowing = ess.Windowing(type='blackmanharris62')
spectrum = ess.Spectrum()
spectralpeaks = ess.SpectralPeaks(orderBy='magnitude',
                                  magnitudeThreshold=0.00001,
                                  minFrequency=20,
                                  maxFrequency=3500,
                                  maxPeaks=60)
hpcp = ess.HPCP()
pool = essentia.Pool()

# Connect streaming algorithms.
loader.audio >> framecutter.signal
framecutter.frame >> windowing.frame >> spectrum.frame
spectrum.spectrum >> spectralpeaks.spectrum
spectralpeaks.magnitudes >> hpcp.magnitudes
spectralpeaks.frequencies >> hpcp.frequencies
hpcp.hpcp >> (pool, 'tonal.hpcp')

# Run streaming network.
essentia.run(loader)

# Detect chords
chord_detection = ChordsDetection(hopSize=2048, windowSize=2)
chords, strengths = chord_detection(essentia.array(pool['tonal.hpcp']))

# Print detected chords and their strengths
for i, (chord, strength) in enumerate(zip(chords, strengths)):
    print(f"Chord {i}: {chord}, Strength: {strength}")