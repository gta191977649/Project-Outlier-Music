# Reverse key map (ensure it's defined before use)
reverse_key_map = {
    0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G",
    8: "G#", 9: "A", 10: "A#", 11: "B"
}


def number_to_chord_label(number):
    normalized_number = ((number - 1) % 12) + 1
    return reverse_key_map.get(normalized_number, "Invalid")


# Function to translate numeral values to chord labels considering minor/major
def translateNumeralValuesToChords(numeral_values):
    chord_array = []

    for value in numeral_values:
        # Determine if the chord is minor or major based on .5
        is_minor = (value % 1) == 0.5
        base_value = int(value)  # Get the integer part for the base chord

        # Get the base chord from the reverse map
        base_chord = reverse_key_map.get(base_value % 12, None)  # Modulo to handle wrapping

        if base_chord:
            # Append the chord with appropriate mode (min or maj)
            mode = "min" if is_minor else "maj"
            chord = f"{base_chord}:{mode}"
            chord_array.append(chord)
        else:
            print(f"Numeral value '{value}' does not correspond to a known chord.")

    return chord_array


# Example usage
numeral_values = [0, 9.5, 8.5, 5, 2, 4.5]  # Some values for testing
result = translateNumeralValuesToChords(numeral_values)
print(result)
