import math

notes = {
    'C': 261.63,  # C4
    'E': 329.63,  # E4
    'G': 392.00  # G4
}

def lcm(f1, f2):
    f1, f2 = int(f1), int(f2)
    return abs(f1 * f2) // math.gcd(f1, f2)

def calculate_subharmonic_period(notes):
    subharmonic_period = 0

    for f1 in notes.values():
        for f2 in notes.values():
            if f1 == f2:
                continue
            lcm_freq = lcm(f1, f2)
            if subharmonic_period == 0 or lcm_freq < subharmonic_period:
                subharmonic_period = lcm_freq

    return subharmonic_period


T_sub = calculate_subharmonic_period(notes)
print(f"(T_sub): {T_sub} Hz")
