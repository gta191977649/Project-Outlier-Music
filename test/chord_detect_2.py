from feature.extract import *
if __name__ == '__main__':
    chords = extractBeatAlignedChordLabels('/Users/nurupo/Desktop/music/c_major.mp3')
    print(chords)