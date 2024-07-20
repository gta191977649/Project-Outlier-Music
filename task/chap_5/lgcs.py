import networkx as nx


def largest_common_subtree(T1, T2):
    def subtree_size(root, G, dp, visited):
        if root in visited:
            return dp[root]

        visited.add(root)
        size = 1
        for child in G[root]:
            size += subtree_size(child, G, dp, visited)

        dp[root] = size
        return size

    def find_largest_common_subtree(r1, r2, G1, G2, dp1, dp2, memo):
        if (r1, r2) in memo:
            return memo[(r1, r2)]

        if r1 is None or r2 is None:
            return 0, []

        size = 0
        nodes = []
        if G1.nodes[r1]['label'] == G2.nodes[r2]['label']:
            size = 1
            nodes = [(r1, G1.nodes[r1]['label'])]
            children1 = list(G1[r1])
            children2 = list(G2[r2])
            for c1 in children1:
                for c2 in children2:
                    child_size, child_nodes = find_largest_common_subtree(c1, c2, G1, G2, dp1, dp2, memo)
                    if child_size > 0:
                        size += child_size
                        nodes.extend(child_nodes)

        memo[(r1, r2)] = (size, nodes)
        return size, nodes

    dp1 = {}
    dp2 = {}
    memo = {}
    visited1 = set()
    visited2 = set()

    root1 = list(nx.topological_sort(T1))[0]
    root2 = list(nx.topological_sort(T2))[0]

    subtree_size(root1, T1, dp1, visited1)
    subtree_size(root2, T2, dp2, visited2)

    size, nodes = find_largest_common_subtree(root1, root2, T1, T2, dp1, dp2, memo)
    return size, nodes


# Example trees
T1 = nx.DiGraph()
T1.add_node(1, label='A')
T1.add_node(2, label='B')
T1.add_node(3, label='C')
T1.add_node(4, label='D')
T1.add_node(5, label='E')
T1.add_node(6, label='F')
T1.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 5), (3, 6)])

T2 = nx.DiGraph()
T2.add_node(1, label='A')
T2.add_node(2, label='B')
T2.add_node(3, label='G')
T2.add_node(4, label='D')
T2.add_node(5, label='E')
T2.add_edges_from([(1, 2), (1, 3), (2, 4), (2, 5)])

size, nodes = largest_common_subtree(T1, T2)

print("Largest Common Subtree Size:", size)
print("Largest Common Subtree Nodes (Label Order):", [label for _, label in nodes])
