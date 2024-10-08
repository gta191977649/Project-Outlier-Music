

def anlysisromanMumerals(chords, is_major=True):
    # Mapping for C major
    major_map = {
        'Cmaj': 'I', 'Dmin': 'ii', 'Emin': 'iii', 'Fmaj': 'IV',
        'Gmaj': 'V', 'Amin': 'vi', 'Bdim': 'vii°'
    }

    chromatic_chord_map_major = {
        'Cmaj': 'I',  # Diatonic in C major
        'C#dim': 'bii°',  # Not diatonic in C major (would be diatonic in C# harmonic/melodic minor)
        'Dmin': 'ii',  # Diatonic in C major
        'D#dim': 'biii°',  # Not diatonic in C major (would be diatonic in Eb harmonic/melodic minor)
        'Emaj': 'III',  # Not diatonic in C major (E is minor in C major)
        'Fmin': 'iv',  # Not diatonic in C major (F is major in C major)
        'Fmaj': 'IV',  # Diatonic in C major
        'F#dim': '#iv°',  # Not diatonic in C major (would be diatonic in F# harmonic/melodic minor)
        'Gmaj': 'V',  # Diatonic in C major
        'G#dim': 'bvi°',  # Not diatonic in C major (would be diatonic in Ab harmonic/melodic minor)
        'Amin': 'vi',  # Diatonic in C major
        'BbMaj': 'bVII',  # Not diatonic in C major (Bb is not in C major scale)
        'Bdim': 'vii°',  # Diatonic in C major
    }

    chromatic_chord_map_minor = {
        'Cmin': 'i',  # Diatonic in C natural minor
        'C#min': 'ii°',  # Not diatonic in C natural minor (would be diatonic in C harmonic/melodic minor)
        'Dmin': 'ii',  # Not diatonic in C natural minor (D is diminished in C natural minor)
        'D#maj': 'III',  # Diatonic in C natural minor
        'Emaj': 'III+',  # Not diatonic in C natural minor (E is diminished in C natural minor)
        'Emin': 'iii',  # Not diatonic in C natural minor (E is diminished in C natural minor)
        'Fmin': 'iv',  # Diatonic in C natural minor
        'Fmaj': 'IV',  # Not diatonic in C natural minor (F is minor in C natural minor)
        'F#min': '#iv°',  # Not diatonic in C natural minor (F# is not in C natural minor scale)
        'Gmin': 'v',  # Diatonic in C natural minor
        'Gmaj': 'V',  # Not diatonic in C natural minor (G is minor in C natural minor)
        'G#maj': 'VI',  # Diatonic in C natural minor
        'Abmaj': 'VI',  # Not diatonic in C natural minor (G# is raised in C harmonic/melodic minor)
        'Amin': 'vii',  # Not diatonic in C natural minor (A is major in C harmonic minor)
        'Amaj': 'VII',  # Not diatonic in C natural minor (A is major in C harmonic minor)
        'Bbmin': 'vii°',  # Not diatonic in C natural minor (Bb is major in C natural minor)
        'Bbmaj': 'bVII',  # Diatonic in C natural minor
        'Bmin': 'vii°',  # Diatonic in C harmonic minor (B natural would be part of the harmonic minor scale)
        'Bmaj': 'VII+',  # Not diatonic in C natural minor (B is diminished in C natural minor)
    }

    # Mapping for A minor
    minor_map = {
        'Amin': 'i', 'Bdim': 'ii°', 'Cmaj': 'III', 'Dmin': 'iv',
        'Emin': 'v', 'Fmaj': 'VI', 'Gmaj': 'VII'
    }

    roman_numerals = []
    # Choose the appropriate mapping based on the key
    chord_map = major_map if is_major else minor_map
    # chord_map = chromatic_chord_map_major if is_major else chromatic_chord_map_minor

    # Convert each chord to its Roman numeral equivalent
    count_non_diatonic = 0
    non_diatonic_chords = []
    for chord in chords:
        roman_numeral = chord_map.get(chord)
        #roman_numerals.append(chord_map.get(chord, "?"))  # Use "?" for unmatched chords
        if roman_numeral is None:
            roman_numerals.append("?")
            non_diatonic_chords.append(chord)
            count_non_diatonic += 1
        else:
            roman_numerals.append(roman_numeral)

    return tuple(roman_numerals), non_diatonic_chords,count_non_diatonic

def is_diatonic_chord( chord,mode="major"):
    major_map = {
        'Cmaj': 'I', 'Dmin': 'ii', 'Emin': 'iii', 'Fmaj': 'IV',
        'Gmaj': 'V', 'Amin': 'vi', 'Bdim': 'vii°'
    }
    minor_map = {
        'Amin': 'i', 'Bdim': 'ii°', 'Cmaj': 'III', 'Dmin': 'iv',
        'Emin': 'v', 'Fmaj': 'VI', 'Gmaj': 'VII'
    }
    chord_map = major_map if mode =="major" else minor_map
    roman_label = chord_map.get(chord)
    return not roman_label is None

def chord_in_list(chord, chords_list):
    return chord in chords_list

def identify_borrowed_chords(progression, mode):
    c_major_chords = ["Cmaj", "Dmin", "Emin", "Fmaj", "Gmaj", "Amin", "Bdim"]
    c_minor_chords = ["Cmin", "Ddim", "Ebmaj", "Fmin", "Gmin", "Abmaj", "Bbmaj"]
    a_minor_chords = ["Amin", "Bdim", "Cmaj", "Dmin", "Emin", "Fmaj", "Gmaj"]
    a_major_chords = ["Amaj", "Bmin", "C#min", "Dmaj", "Emaj", "F#min", "G#dim"]

    borrowed = []

    if mode == "major":
        reference_chords = c_major_chords
        compare_chords = c_minor_chords  # In a major key, borrowed chords come from the parallel minor
    elif mode == "minor":
        reference_chords = a_minor_chords
        compare_chords = a_major_chords  # In a minor key, borrowed chords might come from the parallel major

    for chord_str in progression:
        chord_str = chord_str.replace(":", "")
        if not chord_in_list(chord_str, reference_chords) and chord_in_list(chord_str, compare_chords):
            borrowed.append(chord_str)

    return borrowed
