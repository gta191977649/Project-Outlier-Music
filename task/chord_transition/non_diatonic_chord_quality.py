from feature.analysis import *

if __name__ == '__main__':
    # Example usage:
    key = "C"
    major_chords = ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
    minor_chords = ["Cm", "Ddim", "Eb", "Fm", "Gm", "Ab", "Bb"]
    progression = ["Cmaj", "Cmaj", "Cmaj", "Cmaj", "Cmaj", "Bbmaj", "Bbmaj", "Bbmaj", "Amaj", "Amaj", "Amaj", "Amaj", "Dmin", "Dmin", "Bmin", "Cmaj"]
    roman_numerals,non_diatonic_chords,count_non_diatonic = anlysisromanMumerals(progression,True)
    borrowed_chords = identify_borrowed_chords(progression)
    print("None diatonic chords: ",set(non_diatonic_chords))
    print("Borrowed Chords: ",set(borrowed_chords))