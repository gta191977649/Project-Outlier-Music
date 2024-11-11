import numpy as np
from model.song import Song
import feature.pattern as patternFeature
import feature.beat as beatAnlysis
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer
import seaborn as sns
import matplotlib.pylab as plt
from nltk import ngrams
import re

def normalize_to_max_length(X_train):
    # Find the maximum length of the sequences in X_train
    max_length = max(len(seq) for seq in X_train)

    # Create a new array to hold the normalized sequences
    normalized_X_train = np.zeros((len(X_train), max_length))

    # Fill the new array with the original sequences and pad with zeros
    for i, seq in enumerate(X_train):
        normalized_X_train[i, :len(seq)] = seq

    return normalized_X_train
def isTimmingInSection(time, sectionName, s, matchFirst=True):
    section_timming = None
    for section in s:
        if section["label"] == sectionName:
            section_timming = [float(section['start'])-0.5, float(section['end'])+0.5]
            #print(section_timming)
            if matchFirst: break  # break if we only want to first match
    if section_timming:
        if float(time) >= section_timming[0] and float(time) <= section_timming[1]:
            return True
    return False

"""
NGRAMS Example
Let me explain with an example.

Unigram - [Let] [me] [explain] [with] [an] [example]

Bigram [let me] [me explain] [explain with] [with an] [an example]

Trigram [let me explain] [me explain with] [explain with an] [with an example]
"""
def generateNgrams(sentence,n= 2):
    sentence = ""

def array_to_corpus(X):
    corpus = ""
    for item in X:
        corpus += f"{item} "
    return corpus


def processContour(harmonicRepresentation):
    Contour = []
    for i in range(len(harmonicRepresentation)):
        if i + 1 < len(harmonicRepresentation):
            Contour.append(harmonicRepresentation[i+1] - harmonicRepresentation[i])

    return Contour
if __name__ == '__main__':
    #TARGET_MODE = "major"
    TARGET_SECTION = "chorus"
    #BEATS_PER_BAR = 4
    PROGRESSION_LENGTH = 4#4 chords

    #PATH = "/Users/nurupo/Desktop/dev/audio/europe"
    #PATH = "/Users/nurupo/Desktop/dev/audio/nogizaka46"
    #PATH = "/Users/nurupo/Desktop/dev/audio/akb48"
    #PATH = "E:\\dev\\research\\dataset\\mp3\\haydn"
    #PATH = "/Users/nurupo/Desktop/dev/music4all/akb48"
    PATH = "/Users/nurupo/Desktop/dev/audio/aimyon"
    #PATH = "/Users/nurupo/Desktop/dev/audio/mozart"
    #PATH = "/Users/nurupo/Desktop/dev/audio/haydn"
    print(TARGET_SECTION)
    # loop all folder

    song_collections = []
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith(".h5"):
                filepath = os.path.join(root, file)
                filename = os.path.splitext(file)[0]
                song = Song.from_h5(filepath)
                #if song.mode == TARGET_MODE:
                song_collections.append(song)

    # Prepare pattern dataset
    X_train = []
    Y_songs = []
    Y_chord_progressions = []
    Y_timming = []
    for song in song_collections:
        # for pat in song.chord_pattern:
        #     progression = pat["pattern"]
        #     signal = patternFeature.extractTontalPitchDistancePattern(progression)
        #     X_train.append(signal)

        sections = song.section
        chord_progression = []
        times = []
        x_feature = []
        for i in range(len(song.chord)):
            time, beat, chord = song.chord[i]
            chord = chord.replace(":", "")

            # Todo: ensuring beat alinment is correct
            # if isTimmingInSection(float(time), TARGET_SECTION, sections):
            #     if float(beat) == 1.0 and len(chord_progression) < PROGRESSION_LENGTH:
            #         #print(song.title,time, beat, chord,"YES")
            #         chord_progression.append(chord)
            #         times.append(time)
            if chord == 'N':
                #print("SKIP N Chord")
                continue
            chord_progression.append(chord)

        if len(chord_progression) == 0:
            print(f"Invaild chord progression: {song.title}")
            continue
        #key = f"{song.key}:{song.mode[:3]}"
        # NOPE! We should take the first key from chord progression as home key instead!
        #key = chord_progression[0] # Sets at first chord in progression
        key = f"{song.key}:{song.mode[:3]}"
        #tempoClass = beatAnlysis.getTempoMarking(song.tempo)
        tempoClass = beatAnlysis.getTempoMarkingEncode(song.tempo)
        print(tempoClass)
        print(key)
        signal = patternFeature.extractTontalPitchDistancePattern(chord_progression, key, mode="profile")
        # Get the contour information (simply by using subtract)
        #signal = processContour(signal)
        remove_numbers = {4,0}  # Get rid of perfect / half / tonic cadence
        signal = [num for num in signal if num not in remove_numbers]

        for term in signal:
            x_feature.append([term,tempoClass])

        Y_songs.append(song.title)
        Y_timming.append(times)
        Y_chord_progressions.append(chord_progression)

        # for idx,val in enumerate(signal):
        #     if val == 5.0:
        #         s = patternFeature.extractTontalPitchDistancePattern([chord_progression[idx]], key, mode="profile")
        #         print(s,key,chord_progression[idx])



        X_train.append(array_to_corpus(x_feature))
    print(X_train)


    # Create N-Gram
    stopWords = ([
        "4,5",  # Remove Perfect Cadence
        "0",  # Remove Tonic Case
    ])


    def custom_tokenizer(text):
        pattern = re.compile(r'\[\d+\.\d+, \d+\]')
        matches = pattern.findall(text)
        return matches
    #vectorizer = CountVectorizer(stop_words=None, token_pattern=r"[-]?\d+", ngram_range=(4, 4))
    vectorizer = CountVectorizer(stop_words=None, token_pattern=r'\[\d+\.\d+, \d+\]', ngram_range=(1, 1))
    X = vectorizer.fit_transform(X_train)

    #print(vectorizer.get_feature_names_out())

    tfidf_transformer = TfidfTransformer(use_idf=True)
    train_data = tfidf_transformer.fit_transform(X)
    train_data = train_data.toarray()
    print(train_data.shape)

    # Create a DataFrame for TF-IDF values with feature names
    df_tfidf = pd.DataFrame(train_data, columns=vectorizer.get_feature_names_out())

    # Plotting TF-IDF values using plt.plot
    plt.figure(figsize=(15, 10))
    mean_tfidf_values = df_tfidf.mean().reset_index()
    plt.bar(mean_tfidf_values['index'], mean_tfidf_values[0])
    plt.xticks(rotation=90)
    plt.xlabel('N-grams')
    plt.ylabel('Average TF-IDF Score')
    plt.title(f'{PATH}')
    plt.tight_layout()
    plt.show()

    # IDF weights heat map
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=vectorizer.get_feature_names_out(), columns=["idf_weights"])
    plt.figure(figsize=(10, 8))
    sns.heatmap(train_data, xticklabels=vectorizer.get_feature_names_out())
    plt.xticks(rotation=90)
    plt.xlabel('N-grams')
    plt.title(f'{PATH}')
    plt.tight_layout()
    plt.show()

    # Sort the DataFrame by IDF weights
    df_idf_sorted = df_idf.sort_values(by="idf_weights", ascending=True)
    print(df_idf_sorted)