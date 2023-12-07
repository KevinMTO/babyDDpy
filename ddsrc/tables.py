from ddsrc.edge import Edge, Node


class ComputeTable():
    def lookup(self, x, y):
        # Implement your lookup logic here
        pass

    def insert(self, x, y, result):
        # Implement your insertion logic here
        pass


class UniqueTable:
    def __init__(self):
        self.node_count = 0
        self.collisions = 0
        self.hits = 0
        self.lookups = 0
        self.tables = {}
        self.available = {}  # For reusing nodes

    def get_node(self):
        # A node is available on the stack
        if self.available:
            key, node = self.available.popitem()
            return node

        # New node has to be created
        node = Node(None)

        return node

    def lookup(self, edge, keep_node=False):
        # There are unique terminal nodes
        if edge.next_node is None:
            return edge

        self.lookups += 1
        key = edge.next_node.get_node_hash()
        node_index = edge.next_node.index

        # Successors of a node should either have successive variable numbers or be terminals
        for edge_check in edge.next_node.edges:
            assert edge_check.next_node.index == node_index - 1 or edge_check.next_node.is_terminal()

        node_reference = self.tables.setdefault(node_index, {}).get(key, None)
        while node_reference is not None:
            if edge.next_node.edges == node_reference.edges:
                # Match found
                if edge.next_node is not node_reference and not keep_node:
                    # Put node pointed to by edge.p on available chain
                    self.available[edge.next_node.get_node_hash()] = edge.next_node

                self.hits += 1

                # Variables should stay the same
                assert node_reference.index == edge.next_node.index

                # Successors of a node should either have successive variable
                # numbers or be terminals
                for edge in edge.next_node.edges:
                    assert edge.next_node.index == node_index - 1 or edge.next_node is None

                return Edge(edge.weight, node_reference)

            self.collisions += 1
            node_reference = node_reference.next_unique

        edge.next_node.next_unique = self.tables.setdefault(node_index, {}).get(key, None)
        self.tables.setdefault(node_index, {})[key] = edge.next_node
        self.node_count += 1

        return edge
