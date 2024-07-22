from model.song import Song
import matplotlib.pyplot as plt

if __name__ == '__main__':
    audio_file = r"/mnt/f/music4all/pop/0XTBHHzLg9mngdvU.h5"
    song = Song.from_h5(audio_file)

    # Initialize a dictionary to count label occurrences
    progression_counter = {}

    # Loop through the chords in the song and count labels where beat == 1.0
    print(song.chord)
    for chord in song.chord_transposed:
        time, beat, label = chord
        #if beat == '1.0':
        print(label)
        if label in progression_counter:
            progression_counter[label] += 1
        else:
            progression_counter[label] = 1

    # Plotting the frequency of labels
    labels = list(progression_counter.keys())
    counts = list(progression_counter.values())

    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Chord Labels')
    plt.ylabel('Frequency')
    plt.title('Frequency of Chord Labels with Beat == 1.0')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()