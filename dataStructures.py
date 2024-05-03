class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items)-1]

    def size(self):
        return len(self.items)


class Graph(object):

    edges, nodes = [], []
    adj_matrix = []

    def __init__(self, nodes=[], edges=[]):
        super(Graph, self).__init__()
        self.nodes = nodes
        self.edges = edges
        self.adj_matrix = [
            [0 for i in range(len(nodes))] for j in range(len(nodes))
        ]
        for edge in edges:
            self.adj_matrix[edge[0]-1][edge[1]-1] = 1

    def create_node(self, node_label):
        self.nodes.append(node_label)

    def add_edge(self, edge):
        self.edges.append(edge)

    def print(self):
        print("Nodes: {}, Edges: {}".format(self.nodes, self.edges))
        print("Adjacency matrix:")
        for row in self.adj_matrix:
            print(' '.join(map(lambda x: str(x), row)))
