def detect_secondary_dominants(chord_sequence, mode):
    # Define the diatonic chords and their Roman numerals for C major and C minor
    diatonic_chords = {
        "major": {
            "C:maj": "I", "D:min": "ii", "E:min": "iii", "F:maj": "IV",
            "G:maj": "V", "A:min": "vi", "B:dim": "vii°"
        },
        "minor": {
            "C:min": "i", "D:dim": "ii°", "Eb:maj": "III", "F:min": "iv",
            "G:min": "v", "Ab:maj": "VI", "Bb:maj": "VII"
        }
    }

    # Define the possible secondary dominants
    secondary_dominants = {
        "major": {
            "D:maj": "V/V", "E:maj": "V/vi", "A:maj": "V/ii",
            "B:maj": "V/iii", "C:maj": "V/IV"
        },
        "minor": {
            "C:maj": "V/iv", "D:maj": "V/v", "Eb:maj": "V/VI",
            "F:maj": "V/VII", "G:maj": "V/III"
        }
    }

    roman_numerals = []
    for i, chord in enumerate(chord_sequence):
        # Check if the chord is a potential secondary dominant
        if chord in secondary_dominants[mode]:
            # Look ahead to see if the next chord matches the secondary dominant's target
            if i + 1 < len(chord_sequence):
                next_chord = chord_sequence[i + 1]
                target_chord = secondary_dominants[mode][chord].split('/')[1]
                if next_chord in diatonic_chords[mode] and diatonic_chords[mode][next_chord].lower() == target_chord:
                    roman_numerals.append(secondary_dominants[mode][chord])
                    continue

        # If not a secondary dominant, use the diatonic chord label
        if chord in diatonic_chords[mode]:
            roman_numerals.append(diatonic_chords[mode][chord])
        else:
            # For any non-diatonic chords, use a simple representation
            root, quality = chord.split(':')
            roman_numerals.append(f"{root}({quality})")

    return roman_numerals

# Example usage
chord_sequence = ["C:maj", "D:maj", "G:maj", "C:maj", "A:maj", "D:min", "G:maj", "C:maj"]
mode = "major"

try:
    result = detect_secondary_dominants(chord_sequence, mode)
    print("Input chord sequence:", chord_sequence)
    print("Resulting Roman numerals:", result)
except Exception as e:
    print(f"An error occurred: {e}")
    print("Input chord sequence:", chord_sequence)
    print("Mode:", mode)