import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import matplotlib.lines as mlines
import networkx as nx
import numpy as np


class ChordTransitionPlot:
    def __init__(self,title = "Chord Transition Graph",mode="major"):
        self.mode = mode
        self.title = title
        self.roman_scale_major = ['I', 'V', 'II', 'VI', 'III', 'VII', 'bV', 'bII', 'bVI', 'bIII', 'bVII', 'IV']
        self.roman_scale_minor = ['bIII', 'bVII', 'IV', 'i', 'v', 'ii', 'vi', 'iii', 'vii', 'bV', 'bII', 'bVI']

        self.circle_of_fifths_maj = ['Cmaj', 'Gmaj', 'Dmaj', 'Amaj', 'Emaj', 'Bmaj', 'F#maj', 'C#maj', 'Abmaj', 'Ebmaj',
                                     'Bbmaj', 'Fmaj']
        self.circle_of_fifths_minor = ['Cmin', 'Gmin', 'Dmin', 'Amin', 'Emin', 'Bmin', 'F#min', 'C#min', 'Abmin',
                                       'Ebmin', 'Bbmin', 'Fmin']

        self.enharmonic_map = {
            'G#maj': 'Abmaj',
            'G#min': 'Abmin',
            'D#maj': 'Ebmaj',
            'D#min': 'Ebmin',
            'A#maj': 'Bbmaj',
            'A#min': 'Bbmin',
            'Dbmaj': 'C#maj',
            'C#min': 'Dbmin',
            'Dbmin': 'C#min',
            'Gbmaj': 'F#maj',
            'F#min': 'Gbmin',
            'Gbmin': 'F#min',
        }
        self.circle_of_fifths = self.circle_of_fifths_maj + self.circle_of_fifths_minor
        self.node_pos = {}
        self.label_pos = {}
        self.node_colors = []
        self.node_distance = 2
        self.node_inner_distance = 1.4
        self.node_label_distance = 2.5
        self.chord_node = nx.DiGraph()
        self.chord_scale_label = nx.DiGraph()

        # Initialize node positions
        self.initialize_node_positions()

    def initialize_node_positions(self):
        # Initialize positions for major chords in the outer circle

        if self.mode == "major":
            roman_scale = self.roman_scale_major
        else:  # minor mode
            roman_scale = self.roman_scale_minor

        for i, key in enumerate(roman_scale):
            angle = 2 * np.pi * i / len(roman_scale)  # Evenly space nodes around the circle
            self.label_pos[key] = (self.node_label_distance * np.cos(angle), self.node_label_distance * np.sin(angle))
            self.chord_scale_label.add_node(key)


        for i, key in enumerate(self.circle_of_fifths_maj):
            angle = 2 * np.pi * i / len(self.circle_of_fifths_maj)  # Evenly space nodes around the circle
            self.node_pos[key] = (self.node_distance * np.cos(angle), self.node_distance * np.sin(angle))
            self.chord_node.add_node(key)
            self.node_colors.append(self.is_diatonic(key) and "lightgreen" or "lightgrey")

        for i, key in enumerate(self.circle_of_fifths_minor):
            angle = 2 * np.pi * i / len(self.circle_of_fifths_minor)  # Evenly space nodes around the circle
            self.node_pos[key] = (self.node_inner_distance * np.cos(angle), self.node_inner_distance * np.sin(angle))
            self.chord_node.add_node(key)
            self.node_colors.append(self.is_diatonic(key) and "lightgreen" or "lightgrey")

    def addChordTransition(self, a, b,color="blue"):
        if a == b: return # Skip self connect
        a = a.replace(":","")
        b = b.replace(":","")

        a, b = self.handle_enharmonic(a), self.handle_enharmonic(b)
        if not self.chord_node.has_edge(a, b):
            self.chord_node.add_edge(a, b, weight=0.1,color=color)
        else:
            self.chord_node[a][b]['weight'] += 0.1  # Increase weight for repeated transitions

    def is_diatonic(self,chord_name):
        mode = self.mode
        diatonic_chords_C_major = ['Cmaj', 'Dmin', 'Emin', 'Fmaj', 'Gmaj', 'Amin', 'Bdim']

        diatonic_chords_A_minor = ['Amin', 'Bdim', 'Cmaj', 'Dmin', 'Emin', 'Fmaj', 'Gmaj']

        # Adjust for harmonic minor's V7 chord (E7 in A minor), making G#dim also diatonic in A minor for harmonic purposes
        #diatonic_chords_A_minor_harmonic = diatonic_chords_A_minor + ['Emaj', 'G#dim']
        diatonic_chords_A_minor_harmonic = diatonic_chords_A_minor

        if mode == "major":
            # Check if the chord is diatonic in C major
            return chord_name in diatonic_chords_C_major
        elif mode == "minor":
            # Check if the chord is diatonic in A minor (including harmonic adjustments)
            return chord_name in diatonic_chords_A_minor or chord_name in diatonic_chords_A_minor_harmonic
        else:
            print("Invalid mode specified. Please choose 'major' or 'minor'.")
            return False

    def handle_enharmonic(self, chord):
        mapped_chord = self.enharmonic_map.get(chord, chord)  # Map chord to its enharmonic equivalent if exists
        # Check if the mapped chord exists in the major or minor lists
        if mapped_chord in self.circle_of_fifths_maj or mapped_chord in self.circle_of_fifths_minor:
            return mapped_chord
        else:
            # If the mapped chord doesn't exist, check if the original chord does
            if chord in self.circle_of_fifths_maj or chord in self.circle_of_fifths_minor:
                return chord
            else:
                # Handle the case where neither the mapped nor the original chord exists
                print(f"Warning: Chord {chord} (mapped to {mapped_chord}) does not exist in the defined circles.")
                return None  # or handle differently as needed

    def draw_graph(self, node_pos):
        self.min_edge_width = 0.5
        self.max_edge_width = 15
        self.min_edge_alpha = 0.5
        self.max_edge_alpha = 1



        nx.draw_networkx_nodes(self.chord_node, node_pos, node_color=self.node_colors, edgecolors='black', node_size=1800,node_shape="o")
        nx.draw_networkx_labels(self.chord_node, node_pos)


        nx.draw_networkx_nodes(self.chord_scale_label, self.label_pos, node_color="lightblue", edgecolors='black', node_size=1000,
                               node_shape="s")
        nx.draw_networkx_labels(self.chord_scale_label, self.label_pos)

        all_weights = [data['weight'] for _, _, data in self.chord_node.edges(data=True)]
        min_weight = min(all_weights)
        max_weight = max(all_weights)
        weight_range = max_weight - min_weight if max_weight > min_weight else 1

        for u, v, data in self.chord_node.edges(data=True):
            weight = data['weight']
            edge_color = data.get('color', 'blue')
            norm_weight = (weight - min_weight) / weight_range

            width = self.min_edge_width + norm_weight * (self.max_edge_width - self.min_edge_width)
            alpha = self.min_edge_alpha + norm_weight * (self.max_edge_alpha - self.min_edge_alpha)

            nx.draw_networkx_edges(self.chord_node, node_pos, edgelist=[(u, v)], width=width, edge_color=edge_color,
                                   alpha=alpha, arrowsize=20)

        plt.title(self.title)

    def showPlot(self):
        plt.figure(figsize=(8, 8))
        self.draw_graph(self.node_pos)
        plt.tight_layout()
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')

        # legend
        connection_weight_line = mlines.Line2D([], [], color='blue', marker='_', markersize=15,
                                               label='Diatonic Transition')

        browwed_line = mlines.Line2D([], [], color='red', marker='_', markersize=15,
                                               label='Non Diatonic Transition')
        yellow_line = mlines.Line2D([], [], color='tab:orange', marker='_', markersize=15,
                                               label='Diatonic to Non')

        diatonic_chord_circle = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='lightgreen',
                                              markersize=10, label='Diatonic Chord', linestyle='None')
        nondiatonic_chord_circle = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='lightgrey',
                                                 markersize=10, label='Non-Diatonic Chord', linestyle='None')
        chord_scale_square = mlines.Line2D([], [], color='white', marker='s', markerfacecolor='lightblue',
                                                 markersize=10, label='Chord Degrees', linestyle='None')
        plt.legend(handles=[connection_weight_line, browwed_line,diatonic_chord_circle, yellow_line,nondiatonic_chord_circle,chord_scale_square], loc='best')


        plt.show()
