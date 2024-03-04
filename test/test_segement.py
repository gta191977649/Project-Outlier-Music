from model.song import Song
from feature.section import *
if __name__ == '__main__':
    FILE = "/Users/nurupo/Desktop/dev/music4all/europe/test.mp3"

    s = Song(id="1",title="aa",artist="bbb",file=FILE)
    s.save("/Users/nurupo/Desktop/dev/music4all/europe/test.h5")
    print(s)