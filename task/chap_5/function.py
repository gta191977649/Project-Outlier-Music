from model.song import Song
import matplotlib.pyplot as plt

if __name__ == '__main__':
    audio_file = r"/mnt/f/music4all/pop/0XTBHHzLg9mngdvU.h5"
    song = Song.from_h5(audio_file)

    print(song.chord_change)
    estimate_key = f"{song.key}:{song.mode[:3]}"

    chord_change_sequence = {}
    current = None

    for chord in song.chord_transposed:
        time, beat, label = chord
        if label == "N": continue
        if beat == '1.0': print("------------------")
        print(time, beat, label)
        # if not label == current:
        #     print(time, beat, label)
        # current = label

