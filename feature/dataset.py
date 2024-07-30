import os
from model.song import Song
def loadSongCollection(PATH, mode=None):
    song_collections = []
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith(".h5"):
                filepath = os.path.join(root, file)
                song = Song.from_h5(filepath)
                if mode is None or song.mode == mode:
                    song_collections.append(song)
    return song_collections

def filterRepeatSignal(signal):
    current = None
    out = []
    for x in signal:
        if not x == current:
            out.append(x)
            current = x
    return out

