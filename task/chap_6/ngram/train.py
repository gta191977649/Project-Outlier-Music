import numpy as np
from model.song import Song
import feature.pattern as patternFeature
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer
import seaborn as sns
import matplotlib.pylab as plt
from nltk import ngrams

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, classification_report
import feature.beat as beatAnlysis
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='liblinear')
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

def generateTrainingData(artists):
    # BEATS_PER_BAR = 4
    PROGRESSION_LENGTH = 4  # 4 chords
    X_feature = []
    X_train = []
    Y_label = []
    for artist in artists:
        #TARGET_SECTION = "chorus"

        PATH = f"/Users/nurupo/Desktop/dev/audio/{artist}"
        #PATH = "/Users/nurupo/Desktop/dev/audio/custom"
        #print(TARGET_SECTION)
        # loop all folder

        song_collections = []
        for root, dirs, files in os.walk(PATH):
            for file in files:
                if file.endswith(".h5"):
                    filepath = os.path.join(root, file)
                    filename = os.path.splitext(file)[0]
                    song = Song.from_h5(filepath)
                    # if song.mode == TARGET_MODE:
                    song_collections.append(song)

        # Prepare pattern dataset
        X_artist_song_vector = []
        for song in song_collections:
            sections = song.section
            chord_progression = []
            times = []
            Song_Tokens = ""
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
                    # print("SKIP N Chord")
                    continue
                chord_progression.append(chord)

            if len(chord_progression) == 0:
                print(f"Invaild chord progression: {song.title}")
                continue

            key = f"{song.key}:{song.mode[:3]}"
            tempoClass = beatAnlysis.getTempoMarkingEncode(song.tempo)
            print(key,song.title)
            signal = patternFeature.extractTontalPitchDistancePattern(chord_progression, key, mode="profile")
            remove_numbers = {4, 0}  # Get rid of perfect / half / tonic cadence
            signal = [num for num in signal if num not in remove_numbers]

            for term in signal:
                Song_Tokens = Song_Tokens+f"[{term},{tempoClass}]"
                #Song_Tokens.append([term, tempoClass])
            print(Song_Tokens)
            X_artist_song_vector.append(Song_Tokens)
            Y_label.append(artist)
            X_feature.append(Song_Tokens)

    # done proceess stat for all artists
    print(f"Finish process all artists.")
    stopWords = ([
        "4",  # Remove Perfect Cadence
        "0",  # Remove Tonic Case
    ])
    vectorizer = CountVectorizer(stop_words=None, token_pattern=r'\[\d+\.\d+,\d+\]', ngram_range=(1, 1))
    X = vectorizer.fit_transform(X_feature)
    #print(vectorizer.get_feature_names_out())

    tfidf_transformer = TfidfTransformer(use_idf=True)
    X_tfidf = tfidf_transformer.fit_transform(X)
    X_tfidf = X_tfidf.toarray()
    for item in X_tfidf:
        X_train.append(item)
    # print(X_tfidf)

    return X_train, Y_label



if __name__ == '__main__':
    #TARGET_MODE = "major"
    train_x, y = generateTrainingData([
        "europe",
        "akb48",
        #"nogizaka46",
        #"aimyon",
        #"haydn",
        #"mozart",
    ])
    #print(train_x.shape, y )
    X_train, X_test, y_train, y_test = train_test_split(
        train_x, y, test_size=0.3, random_state=42)
    #print(X_train, X_test, y_train, y_test)
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    recall = recall_score(y_test, predictions, average="macro")
    print(f"recall: {recall}")

    report = classification_report(y_test, predictions, target_names=model.classes_)
    print(report)

    # Cross-validation to better estimate model performance
    scores = cross_val_score(model, train_x, y, cv=5, scoring='recall_macro')
    print(f"Cross-validation recall scores: {scores}")
    print(f"Mean cross-validation recall: {scores.mean()}")