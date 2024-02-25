import feature.extract as feature
import feature.section as section
import feature.analysis as analysis
import feature.pattern as pattern

import h5py
import numpy as np
import json


class Song:
    def __init__(self, id, file, title, artist, key=None, mode=None,tempo=None):
        self.file = file
        # meta data
        self.id = id
        self.title = title
        self.artist = artist
        if not key or not mode:
            key,mode = feature.extract_feature(self.file,feature="key")
        self.key = key
        self.mode = mode
        self.tempo = tempo or feature.extract_feature(self.file,feature="tempo")
        self.album = ""
        self.release = ""
        self.language = ""
        self.tags = ""
        self.chord = feature.extractBeatAlignedChordLabels(self.file)
        # Calculate transpose amount
        self.transpose_amount = feature.calculate_transpose_amount(self.key, self.mode)
        self.chord_transposed = []
        if self.transpose_amount == 0:
            print(f"Skip transpose for {title}, since it's already on standard!")
            self.chord_transposed = self.chord
        else:
            self.chord_transposed = feature.transposeBeatAlignedChordLabels(self.chord, self.transpose_amount)
        #self.section = section.extractSongSection(file)
        self.section = []
        # calculate chord summary patterns
        self.chord_pattern = pattern.summaryChordPattern(self.chord_transposed)
        self.chord_change = pattern.extractChangeChordPattern(self.chord_transposed)
        # append annotation
        isMajor = self.mode == "major" and True or False
        roman_numerals, non_diatonic_chords, non_diatonic_count = analysis.anlysisromanMumerals(
            self.chord_change["valid_sequence"],
            isMajor)
        self.chord_change["roman_label"] = roman_numerals
        self.chord_change["non_diatonic_chords"] = non_diatonic_chords
        self.chord_change["non_diatonic_chords_count"] = non_diatonic_count

        # append diatonic analysis
        for ptn in self.chord_pattern:
            roman_numerals, non_diatonic_chords,non_diatonic_count = analysis.anlysisromanMumerals(ptn["pattern"], isMajor)
            ptn["roman_label"] = roman_numerals
            ptn["non_diatonic_chords"] = non_diatonic_chords
            ptn["non_diatonic_chords_count"] = non_diatonic_count * int(ptn["matches"])



    def save(self, path):
        # save parsed feature to file
        with h5py.File(path, "w") as f:
            # save meta
            g_meta = f.create_group("metadata")
            g_meta.create_dataset("id", data=self.id)
            g_meta.create_dataset("title", data=self.title)
            g_meta.create_dataset("artist", data=self.artist)
            g_meta.create_dataset("key", data=self.key)
            g_meta.create_dataset("mode", data=self.mode)
            g_meta.create_dataset("tempo", data=self.tempo)
            g_meta.create_dataset("transpose_amount", data=self.transpose_amount)

            album_data = self.album if self.album is not None else ""
            release_data = self.release if self.release is not None else ""
            g_meta.create_dataset("album", data=album_data)
            g_meta.create_dataset("release", data=release_data)

            # save chords
            g_chord = f.create_group("chord")
            chord_array = np.array(self.chord, dtype='S')
            chord_transposed_array = np.array(self.chord_transposed, dtype='S')
            g_chord.create_dataset("chord_original", data=chord_array)
            g_chord.create_dataset("chord_transposed", data=chord_transposed_array)
            serialized_chord_pattern = [json.dumps(d) for d in self.chord_change]
            chord_change_array = np.array(serialized_chord_pattern, dtype='S')
            g_chord.create_dataset("chord_changes", chord_change_array)

            # save chord pattern
            g_pattern = f.create_group("pattern")
            serialized_chord_pattern = [json.dumps(d) for d in self.chord_pattern]
            chord_pattern_array = np.array(serialized_chord_pattern, dtype='S')
            g_pattern.create_dataset("chord_pattern", data=chord_pattern_array)

            # save section label
            g_section = f.create_group("section")
            section_array = np.array(self.section, dtype='S')
            g_section.create_dataset("section_label", data=section_array)

        print(f"✅{path} saved!")
