from chordparser import Parser
# Initialize the parser
cp = Parser()

# Define a key
key = 'C'

# Define a chord progression using chord symbols
chord_symbols = ['C', 'Dm', 'Em', 'F', 'G', 'Am', 'Bdim']

# Analyze and print Roman numerals for each chord symbol
for symbol in chord_symbols:
    new_chord = cp.create_chord(symbol)
    rn = cp.to_roman(new_chord, cp.create_scale(key, "major"))
    print(f'{symbol}: {rn}')
