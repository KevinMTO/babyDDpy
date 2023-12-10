class Node:
    def __init__(self, index):
        self.index = index
        self.edges = []
        self.father = None
        self.terminal = False
        self.dd_hash = None
        self.data_cache = None
        self.identity = None
        self.next_unique = None

    def set_terminal_up(self):
        self.terminal = True

    def set_node_hash(self):
        successor_hashes = [e.next_node.dd_hash for e in self.edges]
        # Replace with an appropriate hash function for your use case
        self.dd_hash = hash((self.index, tuple(successor_hashes)))
        return self.dd_hash

    def get_node_hash(self):
        if self.dd_hash is None:
            return self.set_node_hash()
        else:
            return self.dd_hash

    def is_terminal(self):
        return self.terminal


zero = Node("zero")
zero.set_terminal_up()
zero.set_node_hash()
one = Node("one")
one.set_terminal_up()
one.set_node_hash()


class Edge:
    def __init__(self, weight=0. + 0.j, next_node=None, father_node=None):
        self.weight = weight
        self.next_node = next_node
        self.father_node = father_node
        self.isterminal = True if next_node == one or next_node == zero else False

    def is_terminal(self):
        return self.isterminal

    @classmethod
    def one(cls):
        return cls(1 + 0.j, one)

    @classmethod
    def zero(cls):
        return cls(0. + 0.j, zero)

    def set_father_node(self, node):
        self.father_node = node

    def copy(self):
        # Create a new instance of Edge with the same attributes
        copied_edge = type(self)(self.weight, self.next_node, self.father_node)
        copied_edge.isterminal = self.isterminal
        return copied_edge


class MEdge(Edge):
    def __init__(self, weight=0. + 0.j, next_node=None, father_node=None):
        super().__init__(weight=weight, next_node=next_node, father_node=father_node)


class VEdge(Edge):
    def __init__(self, weight=0. + 0.j, next_node=None, father_node=None):
        super().__init__(weight=weight, next_node=next_node, father_node=father_node)
