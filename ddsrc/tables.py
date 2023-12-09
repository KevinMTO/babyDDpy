from ddsrc.edge_nodes import Edge, Node


class ComputeTable():
    def __init__(self):
        self.table = {}

    def insert(self, left_operand: Edge, right_operand: Edge, result):
        key = hash((left_operand.next_node.get_node_hash(), right_operand.next_node.get_node_hash()))
        self.table[key] = {'left_operand': left_operand, 'right_operand': right_operand, 'result': result}

    def lookup(self, left_operand: Edge, right_operand: Edge):
        result_type = type(right_operand)
        result = result_type()

        key = hash((left_operand.next_node.get_node_hash(), right_operand.next_node.get_node_hash()))
        entry = self.table.get(key, {})

        if entry.get('result', None) is None:
            return result
        if entry.get('left_operand') != left_operand or entry.get('right_operand') != right_operand:
            return result

        return entry['result']


class AdditionTable(ComputeTable):
    def __init__(self):
        super().__init__()


class MultTable(ComputeTable):
    def __init__(self):
        super().__init__()


class UniqueTable:
    def __init__(self):
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

                # Variables should stay the same
                assert node_reference.index == edge.next_node.index

                # Successors of a node should either have successive variable
                # numbers or be terminals
                for edge in edge.next_node.edges:
                    assert edge.next_node.index == node_index - 1 or edge.next_node is None

                return Edge(edge.weight, node_reference)

            node_reference = node_reference.next_unique

        edge.next_node.next_unique = self.tables.setdefault(node_index, {}).get(key, None)
        self.tables.setdefault(node_index, {})[key] = edge.next_node

        return edge
