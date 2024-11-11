import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from feature.chord import *
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from collections import defaultdict
import networkx as nx
from ete3 import Tree, TreeStyle, NodeStyle, TextFace
import colorsys
import plotly.graph_objects as go
import igraph as ig
from sklearn.manifold import TSNE
import umap
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn import manifold

class ChordProgressionIsolationForest:
    def __init__(self, chord_progressions,tokenMethod="numeral",dimReduceMethod=None):
        self.tokenMethod = tokenMethod
        self.dimReduceMethod = dimReduceMethod
        self.chord_progressions = chord_progressions
        self.vectorized_progressions,self.feature_names = self._vectorize_progressions(method=tokenMethod)
        self.vectorized_progressions_original = self.vectorized_progressions.copy()
        self.vectorized_progressions = self._reduceDimensionality(method=dimReduceMethod)

        self.outlier_detector = IsolationForest(n_estimators=500,contamination=0.1, random_state=42)
        self.outliers = self.detect_outliers()

    def _vectorize_progressions(self, method="numeral"):
        if method == "numeral":
            if self.chord_progressions.ndim == 1:
                tokenlized = extractChordNumeralValues(self.chord_progressions)
                output = []
                for token in tokenlized:
                    output.append([token])
                return output,[]
            else:
                return np.array([
                    extractChordNumeralValues(progression)
                    for progression in self.chord_progressions
                ]),[]
        elif method == "onehot":
            #flattened = np.array(self.chord_progressions).flatten().reshape(-1, 1)
            if self.chord_progressions.ndim == 1:
                encoder = OneHotEncoder(sparse=False)
                onehot = encoder.fit_transform(self.chord_progressions.reshape(-1,1))
                return onehot,encoder.get_feature_names_out()
            else:
                encoder = OneHotEncoder(sparse=False)
                onehot = encoder.fit_transform(self.chord_progressions)
                return onehot.reshape(len(self.chord_progressions), -1),encoder.get_feature_names_out()
        elif method == "ordinal":
            flattened = np.array(self.chord_progressions).flatten().reshape(-1, 1)
            encoder = OrdinalEncoder()
            onehot = encoder.fit_transform(flattened)
            return onehot.reshape(len(self.chord_progressions), -1),encoder.get_feature_names_out()
        elif method == "direct":
            return self.chord_progressions
        else:
            raise ValueError("Unsupported vectorization method")

    def _reduceDimensionality(self,method=None):
        if method == None: return self.vectorized_progressions
        elif method == "t-sne":
            tsne = TSNE(perplexity=30,n_components=2,metric="cosine", n_iter=1000)
            tsne_result = tsne.fit_transform(self.vectorized_progressions)
            return tsne_result
        elif method == "umap":
            umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            umap_embedding = umap_reducer.fit_transform(self.vectorized_progressions)
            return umap_embedding
    def detect_outliers(self):
        #self.outlier_detector.fit(self.vectorized_progressions)
        return self.outlier_detector.fit_predict(self.vectorized_progressions)

    def compute_anomaly_scores(self):
        return -self.outlier_detector.score_samples(self.vectorized_progressions)

    def visualize_anomaly_scores(self):
        anomaly_scores = self.compute_anomaly_scores()

        fig, ax = plt.subplots(figsize=(5, 2))

        x = np.arange(len(self.chord_progressions))

        # Create a list of colors based on outlier status
        colors = ['r' if outlier == -1 else 'b' for outlier in self.outliers]

        # Plot the bars with colors based on outlier status
        bars = ax.bar(x, anomaly_scores, color=colors, alpha=0.7)

        ax.set_ylabel('Anomaly Score')
        ax.set_xlabel('Chord Progressions')
        ax.set_title('Anomaly Scores for Chord Progressions (Red: Outliers)')
        ax.set_xticks(x)

        # Add a legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.7, label='Outlier'),
                           plt.Rectangle((0, 0), 1, 1, color='blue', alpha=0.7, label='Normal')]
        ax.legend(handles=legend_elements, loc='upper right')

        # Rotate x-axis labels if needed
        # plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

    def print_outliers(self):
        outliers = self.outliers
        anomaly_scores = self.compute_anomaly_scores()

        print("Outliers detected:")
        for i, (is_outlier, score) in enumerate(zip(outliers, anomaly_scores)):
            if is_outlier == -1:  # -1 indicates an outlier in Isolation Forest
                print(f"Progression {i}: {self.chord_progressions[i]}")
                print(f"Anomaly Score: {score}")
                print("--------------------")
    def save_outliers(self,path):
        outliers = self.outliers
        anomaly_scores = self.compute_anomaly_scores()
        output = {
            "index":[],
            "outlier":[],
            "score":[],
        }

        print("Outliers detected:")
        for i, (is_outlier, score) in enumerate(zip(outliers, anomaly_scores)):
            if is_outlier == -1:  # -1 indicates an outlier in Isolation Forest
                output["index"].append(i)
                output["outlier"].append(self.chord_progressions[i])
                output["score"].append(score)

        csv = pd.DataFrame(output)
        print(csv)
        csv.to_csv(path)
    def plot_result(self,annOutlier=True):

        if self.dimReduceMethod == None:
            if self.tokenMethod == "onehot":
                fig = plt.figure(figsize=(12, 14))
                gs = GridSpec(2, 1, height_ratios=[5, 1], figure=fig)

                # Heatmap subplot
                ax_heatmap = fig.add_subplot(gs[0])
                data = self.vectorized_progressions.copy()
                sns.heatmap(data, cmap="binary_r", cbar=False, xticklabels=self.feature_names, ax=ax_heatmap)

                if annOutlier:
                    # Highlight outlier rows
                    for idx, is_outlier in enumerate(self.outliers):
                        if is_outlier == -1:  # -1 indicates an outlier
                            ax_heatmap.add_patch(plt.Rectangle((0, idx), data.shape[1], 1,
                                                               fill=True, facecolor='red', alpha=0.5))

                ax_heatmap.set_title("One-Hot Encoded Chord Progressions (Red overlay: Outliers)")

                # Chord frequency subplot
                ax_freq = fig.add_subplot(gs[1], sharex=ax_heatmap)

                # Calculate the frequency of each chord
                chord_frequencies = data.sum(axis=0) / data.shape[0]

                # Plot bar chart of chord frequencies
                ax_freq.bar(range(len(chord_frequencies)), chord_frequencies)

                ax_freq.set_ylabel("Frequency")
                ax_freq.set_xlim(0, len(self.feature_names) - 1)

                # Set x-axis labels
                ax_freq.set_xticks(range(len(self.feature_names)))
                ax_freq.set_xticklabels(self.feature_names, rotation=90)

                # Adjust layout
                plt.tight_layout()
                plt.show()
        else: # if used any dimemtional reduce
            for idx, point in enumerate(self.vectorized_progressions):
                if self.outliers[idx] == -1:
                    plt.scatter(x=point[0], y=point[1], color="red")
                else:
                    plt.scatter(x=point[0], y=point[1], color="blue")
            plt.show()
            # heatmap

            plt.figure(figsize=(10, 10))
            # Create a copy of the data
            data = self.vectorized_progressions_original

            # Plot the heatmap for all data
            ax = sns.heatmap(data, cmap="binary_r", cbar=False,
                             xticklabels=self.feature_names)

            # Highlight outlier rows
            for idx, is_outlier in enumerate(self.outliers):
                if is_outlier == -1:  # -1 indicates an outlier
                    ax.add_patch(plt.Rectangle((0, idx), data.shape[1], 1,
                                               fill=True, facecolor='red', alpha=0.5))

            plt.xticks(rotation=90)
            plt.title("One-Hot Encoded Chord Progressions (Red overlay: Outliers)")

            plt.tight_layout()
            plt.show()

    def visualize_progression_tree_hierarchical(self, outliers_only=False):
        # Create a tree structure
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        outliers = self.detect_outliers()
        anomaly_scores = self.compute_anomaly_scores()

        for i, (prog, is_outlier, score) in enumerate(zip(self.chord_progressions, outliers, anomaly_scores)):
            if not outliers_only or (outliers_only and is_outlier):
                tree[prog[0]][prog[1]][prog[2]][prog[3]].append((i, is_outlier, score))

        # Calculate the maximum count for normalization
        max_count = max(len(leaves) for level1 in tree.values() for level2 in level1.values()
                        for level3 in level2.values() for leaves in level3.values())

        # Create ETE Tree
        ete_tree = self._build_ete_tree(tree, max_count, outliers_only)

        # Create TreeStyle
        ts = TreeStyle()
        ts.show_leaf_name = False
        ts.rotation = 90
        ts.show_scale = False
        ts.mode = "c"

        ts.layout_fn = lambda node: self._custom_layout(node, outliers_only)
        ts.branch_vertical_margin = 15
        title = "Chord Progression Tree (Outliers Only)" if outliers_only else "Chord Progression Tree (Red: Outliers, Green: Common Patterns)"
        ts.title.add_face(TextFace(title, fsize=20), column=0)

        # Render tree
        ete_tree.show(tree_style=ts)
        ete_tree.render("mytree.png", w=183, units="mm")

        # If you want to save the tree to a file instead of showing it:
        #ete_tree.render("chord_progression_tree.png", tree_style=ts, w=1200, h=800, units="px")

    def _build_ete_tree(self, node, max_count, outliers_only=False, parent=None):
        if isinstance(node, list):
            count = len(node)
            is_outlier = any(o == -1 for _, o, _ in node)
            avg_score = np.mean([s for _, o, s in node if o == -1]) if is_outlier else 0
            frequency = count / max_count

            if outliers_only and not is_outlier:
                return None

            name = f"{parent.name}|{count}|{is_outlier}|{avg_score:.4f}|{frequency:.4f}"
            return Tree(name=name)

        if parent is None:
            tree = Tree(name="Root")
        else:
            tree = Tree(name=parent.name)

        for child_name, child_node in node.items():
            child_tree = self._build_ete_tree(child_node, max_count, outliers_only, Tree(name=child_name))
            if child_tree is not None:
                tree.add_child(child_tree)

        return tree if tree.children else None

    def _custom_layout(self, node, outliers_only=False):
        if node.is_leaf():
            chord, count, is_outlier, avg_score, frequency = node.name.split("|")
            count = int(count)
            is_outlier = is_outlier == "True"
            avg_score = float(avg_score)
            frequency = float(frequency)

            if outliers_only and not is_outlier:
                return

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
            textfont=dict(family="MS PGothic",size=font_size),
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

    def visualize_progression_dendrogram(self, outliers_only=False):
        # Create a tree structure
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        outliers = self.detect_outliers()
        anomaly_scores = self.compute_anomaly_scores()

        # Flatten the chord progressions and create labels
        flat_progressions = []
        labels = []
        for i, (prog, is_outlier, score) in enumerate(zip(self.chord_progressions, outliers, anomaly_scores)):
            if not outliers_only or (outliers_only and is_outlier):
                flat_progressions.append(prog)
                label = ' -> '.join(prog)
                if is_outlier:
                    label += f' (Outlier, Score: {score:.2f})'
                labels.append(label)

        # Convert chord progressions to numerical data
        # This is a simple conversion, you might want to use a more sophisticated method
        unique_chords = list(set(chord for prog in flat_progressions for chord in prog))
        chord_to_num = {chord: i for i, chord in enumerate(unique_chords)}
        numerical_data = [[chord_to_num[chord] for chord in prog] for prog in flat_progressions]

        # Perform hierarchical clustering
        linked = linkage(numerical_data, 'ward')

        # Create figure and axis
        plt.figure(figsize=(15, 10))
        ax = plt.gca()

        # Plot dendrogram
        dendrogram(linked,
                   orientation='right',
                   labels=labels,
                   leaf_font_size=8,
                   ax=ax)

        # Customize the plot
        ax.set_title('Chord Progression Dendrogram' + (' (Outliers Only)' if outliers_only else ''))
        ax.set_xlabel('Distance')
        ax.set_ylabel('Chord Progressions')

        # Rotate labels for better readability
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

        # Adjust layout and display
        plt.tight_layout()
        plt.show()

        # If you want to save the dendrogram to a file:
        # plt.savefig("chord_progression_dendrogram.png", dpi=300, bbox_inches='tight')