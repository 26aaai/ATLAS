import networkx as nx
import numpy as np

class TopologicalMap:
    def __init__(self):
        self.graph = nx.Graph()
        self.node_features = {}  # {node_id: feature}
        self.visited_nodes = set()
        self.current_node = None

    def add_node(self, node_id, feature, is_current=False):
        self.graph.add_node(node_id)
        self.node_features[node_id] = feature
        if is_current:
            self.current_node = node_id
            self.visited_nodes.add(node_id)

    def add_edge(self, node_id1, node_id2):
        self.graph.add_edge(node_id1, node_id2)

    def update(self, obs):
        node_id = obs['current_node_id']
        feature = obs['current_feature']
        self.add_node(node_id, feature, is_current=True)
        for neighbor in obs.get('neighbor_nodes', []):
            n_id = neighbor['node_id']
            n_feat = neighbor['feature']
            self.add_node(n_id, n_feat)
            self.add_edge(node_id, n_id)
        if 'STOP' not in self.graph.nodes:
            stop_feat = np.zeros_like(feature)
            stop_feat[-4:] = [0,0,0,1]
            self.add_node('STOP', stop_feat)

    def get_graph_data(self):
        node_ids = list(self.graph.nodes)
        features = [self.node_features[nid] for nid in node_ids]
        adj = nx.to_numpy_array(self.graph, nodelist=node_ids)
        return node_ids, features, adj
