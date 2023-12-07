from ddsrc.edge import Edge


class VEdge(Edge):
    def __init__(self, weight, next_node, father_node=None):
        super().__init__(weight=weight, next_node=next_node, father_node=father_node)


