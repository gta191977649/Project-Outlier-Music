import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from feature.chord import *
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
import networkx as nx
from ete3 import Tree, TreeStyle, NodeStyle, TextFace
import colorsys
import plotly.graph_objects as go
import igraph as ig

class ChordProgressionIsolationForest:
    def __init__(self, chord_progressions,tokenMethod="numeral"):
        self.chord_progressions = chord_progressions
        self.vectorized_progressions = self._vectorize_progressions(method=tokenMethod)
        self.outlier_detector = IsolationForest(n_estimators=500,contamination=0.1, random_state=0)
        print(self.vectorized_progressions)

    def _vectorize_progressions(self, method="numeral"):
        if method == "numeral":
            return np.array([
                extractChordNumeralValues(progression)
                for progression in self.chord_progressions
            ])
        elif method == "onehot":
            flattened = np.array(self.chord_progressions).flatten().reshape(-1, 1)
            encoder = OneHotEncoder(sparse=False)
            onehot = encoder.fit_transform(flattened)
            return onehot.reshape(len(self.chord_progressions), -1)
        else:
            raise ValueError("Unsupported vectorization method")

    def detect_outliers(self):
        self.outlier_detector.fit(self.vectorized_progressions)
        return self.outlier_detector.predict(self.vectorized_progressions)

    def compute_anomaly_scores(self):
        return -self.outlier_detector.score_samples(self.vectorized_progressions)

    def visualize_anomaly_scores(self):
        anomaly_scores = self.compute_anomaly_scores()

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(self.chord_progressions))

        ax.bar(x, anomaly_scores, color='green', alpha=0.7)
        ax.set_ylabel('Anomaly Score')
        ax.set_xlabel('Chord Progressions')
        ax.set_title('Anomaly Scores for Chord Progressions')
        ax.set_xticks(x)
        #ax.set_xticklabels(['-'.join(prog) for prog in self.chord_progressions], rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

    def print_outliers(self):
        outliers = self.detect_outliers()
        anomaly_scores = self.compute_anomaly_scores()

        print("Outliers detected:")
        for i, (is_outlier, score) in enumerate(zip(outliers, anomaly_scores)):
            if is_outlier == -1:  # -1 indicates an outlier in Isolation Forest
                print(f"Progression {i}: {self.chord_progressions[i]}")
                print(f"Anomaly Score: {score}")
                print("--------------------")

    def visualize_progression_tree_hierarchical(self):
        # Create a tree structure
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        outliers = self.detect_outliers()
        anomaly_scores = self.compute_anomaly_scores()

        for i, (prog, is_outlier, score) in enumerate(zip(self.chord_progressions, outliers, anomaly_scores)):
            tree[prog[0]][prog[1]][prog[2]][prog[3]].append((i, is_outlier, score))

        # Calculate the maximum count for normalization
        max_count = max(len(leaves) for level1 in tree.values() for level2 in level1.values()
                        for level3 in level2.values() for leaves in level3.values())

        # Create ETE Tree
        ete_tree = self._build_ete_tree(tree, max_count)

        # Create TreeStyle
        ts = TreeStyle()
        ts.show_leaf_name = False
        ts.show_scale = False

        ts.layout_fn = self._custom_layout
        ts.branch_vertical_margin = 15
        ts.title.add_face(TextFace("Chord Progression Tree (Red: Outliers, Green: Common Patterns)", fsize=20),
                          column=0)

        # Render tree
        ete_tree.show(tree_style=ts)
        ete_tree.render("mytree.png", w=183, units="mm")

        # If you want to save the tree to a file instead of showing it:
        # ete_tree.render("chord_progression_tree.png", tree_style=ts, w=1200, h=800, units="px")
    def _build_ete_tree(self, node, max_count, parent=None):
        if isinstance(node, list):
            count = len(node)
            is_outlier = any(o == -1 for _, o, _ in node)
            avg_score = np.mean([s for _, o, s in node if o == -1]) if is_outlier else 0
            frequency = count / max_count
            name = f"{parent.name}|{count}|{is_outlier}|{avg_score:.4f}|{frequency:.4f}"
            return Tree(name=name)

        if parent is None:
            tree = Tree(name="Root")
        else:
            tree = Tree(name=parent.name)

        for child_name, child_node in node.items():
            child_tree = self._build_ete_tree(child_node, max_count, Tree(name=child_name))
            tree.add_child(child_tree)

        return tree
    def _custom_layout(self, node):
        if node.is_leaf():
            chord, count, is_outlier, avg_score, frequency = node.name.split("|")
            count = int(count)
            is_outlier = is_outlier == "True"
            avg_score = float(avg_score)
            frequency = float(frequency)

            node_style = NodeStyle()
            node_style["shape"] = "sphere"
            node_style["size"] = 10 + count * 2  # Increase size based on count

            if is_outlier:
                node_style["fgcolor"] = "red"
            else:
                # Use a color gradient from light blue (low frequency) to dark green (high frequency)
                hue = 0.3 + (1 - frequency) * 0.3  # Range from 0.3 (green) to 0.6 (blue)
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
                node_style["fgcolor"] = f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"

            node.set_style(node_style)

            if is_outlier:
                text = f"{chord}\nCount: {count}\nScore: {avg_score:.2f}"
            else:
                text = f"{chord}\nCount: {count}"

            text_face = TextFace(text, fgcolor="black", fsize=8)
            node.add_face(text_face, column=0, position="branch-right")
        else:
            node_style = NodeStyle()
            node_style["shape"] = "sphere"
            node_style["size"] = 8
            node_style["fgcolor"] = "gray"

            node.set_style(node_style)

            text_face = TextFace(node.name, fgcolor="black", fsize=8)
            node.add_face(text_face, column=0, position="branch-right")


    def visualize_in_web(self,horizontal_spacing=1.0,node_size_scale=1.0,font_size=10):
        # Create a tree structure
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        outliers = self.detect_outliers()
        anomaly_scores = self.compute_anomaly_scores()

        for i, (prog, is_outlier, score) in enumerate(zip(self.chord_progressions, outliers, anomaly_scores)):
            tree[prog[0]][prog[1]][prog[2]][prog[3]].append((i, is_outlier, score))

        # Create igraph tree
        g, labels, colors, sizes, hover_texts = self._create_igraph_tree(tree, node_size_scale)

        # Use igraph's tree layout
        layout = g.layout_reingold_tilford(mode="out")

        # Extract positions
        Xn, Yn = zip(*layout)

        # Normalize positions
        Yn = [-y for y in Yn]  # Flip y-axis to have root at the top
        max_y = max(Yn)
        min_y = min(Yn)
        if max_y != min_y:
            Yn = [(y - min_y) / (max_y - min_y) for y in Yn]
        else:
            Yn = [0 for _ in Yn]  # All nodes at the same level

        # Apply horizontal spacing
        max_x = max(Xn)
        min_x = min(Xn)
        if max_x != min_x:
            Xn = [(x - min_x) / (max_x - min_x) * horizontal_spacing for x in Xn]
        else:
            Xn = [0.5 * horizontal_spacing for _ in Xn]  # All nodes in a single column

        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in g.es:
            x0, y0 = Xn[edge.source], Yn[edge.source]
            x1, y1 = Xn[edge.target], Yn[edge.target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        # Create node trace
        node_trace = go.Scatter(
            x=Xn, y=Yn,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color=colors,
                size=sizes,
                line_width=2),
            text=labels,
            textposition="top center",
            textfont=dict(size=font_size),
            hovertext=hover_texts)

        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Chord Progression Tree',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))

        # Adjust the plot width based on horizontal spacing
        fig.update_layout(width=max(600, int(600 * horizontal_spacing)))

        fig.show()

    def _create_igraph_tree(self, tree, node_size_scale):
        g = ig.Graph(directed=True)
        root = g.add_vertex(name="Root")
        labels = ["Root"]
        colors = ["lightgrey"]
        sizes = [20]
        hover_texts = ["Root"]

        def add_nodes(current_dict, parent_vertex):
            for key, value in current_dict.items():
                if isinstance(value, list):
                    count = len(value)
                    is_outlier = any(o == -1 for _, o, _ in value)
                    avg_score = sum(s for _, o, s in value if o == -1) / count if is_outlier else 0

                    v = g.add_vertex(name=key)
                    g.add_edge(parent_vertex, v)

                    labels.append(key)
                    colors.append("red" if is_outlier else f"rgb(0, {min(255, count * 10)}, 0)")
                    sizes.append(max(5, min(50, 5 + count * node_size_scale)))

                    hover_text = f"Progression: {key}\nCount: {count}\n"
                    hover_text += "Outlier\n" if is_outlier else "Normal\n"
                    if is_outlier:
                        hover_text += f"Avg Score: {avg_score:.2f}"

                    hover_texts.append(hover_text)
                else:
                    v = g.add_vertex(name=key)
                    g.add_edge(parent_vertex, v)

                    labels.append(key)
                    colors.append("lightblue")
                    sizes.append(10)
                    hover_texts.append(f"Partial Progression: {key}")

                    add_nodes(value, v)

        add_nodes(tree, root)
        return g, labels, colors, sizes, hover_texts