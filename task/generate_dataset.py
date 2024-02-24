import feature.extract as feature
from model.song import Song

if __name__ == '__main__':
    file = "/Users/nurupo/Desktop/dev/Project-Outlier-Music/dataset/music4all/data/a.mp3"
    id = 1
    title = "test"
    artist = "test"
    album = "test"
    release = 2000
    key = "C"
    mode = "major"
    song = Song(id=id,file=file, title=title, artist=artist,key=key,mode=mode)
    song.save("/Users/nurupo/Desktop/dev/Project-Outlier-Music/dataset/music4all/data/a.h5")
    print(song)
