from pychord import Chord
from feature.harmonic import note_frequency,find_harmonic_tension
class Harmonic:
    def __init__(self,chord_name):
        chord_name = chord_name.replace(":","")
        self.chord = Chord(chord_name)
        self.chord_notes = self.chord.components()
        self.chord_frequencies = [note_frequency(note,4) for note in self.chord_notes]
        self.tension,self.alignment_times,self.chord_signal = find_harmonic_tension(self.chord_frequencies)
