import os
from model.song import Song
def loadSongCollection(PATH, filter=None):
    song_collections = []
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith(".h5"):
                filepath = os.path.join(root, file)
                song = Song.from_h5(filepath)
                if filter is None or song.mode == filter:
                    song_collections.append(song)
    return song_collections