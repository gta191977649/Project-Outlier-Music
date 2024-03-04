from feature.analysis import *
from model.song import *
if __name__ == '__main__':
    h5 = "/Users/nurupo/Desktop/dev/music4all/audios/0GOUjjayzUo2a0Jt.h5"
    song = Song.from_h5(h5)

    progression = ["C", "Dm", "Eb", "F", "G", "Am", "Bdim"]

    b = identify_borrowed_chords(song.chord_pattern[0]["pattern"],"major")
    print(b)