import numpy as np

from ddsrc.edge_nodes import MEdge, VEdge
tolerance = 1e-16


class VEdgeNormalization:
    """
    def normalize(edge: VEdge) -> VEdge:
        sum_norm2 = 0 #np.abs(edge.next_node.edges[0].weight) ** 2
        mag2_max = 0 #np.abs(edge.next_node.edges[0].weight) ** 2
        arg_max = 0

        for i in range(0, len(edge.next_node.edges)):
            sum_norm2 += np.abs(edge.next_node.edges[i].weight) ** 2

        for i in range(0, len(edge.next_node.edges)):
            if np.abs(edge.next_node.edges[i].weight) + tolerance >= mag2_max:
                mag2_max = np.abs(edge.next_node.edges[i].weight) ** 2
                arg_max = i

        norm = np.sqrt(sum_norm2)
        mag_max = np.sqrt(mag2_max)
        common_factor = norm #/ mag_max

        #edge.weight = edge.next_node.edges[arg_max].weight
        real_part = np.real(edge.weight) * common_factor
        img_part = np.imag(edge.weight) * common_factor
        edge.weight = np.complex128(real_part + 1j * img_part)

        if np.abs(edge.weight) < tolerance:
            return VEdge.zero()

        #if mag_max == 0. or (mag_max / norm) < tolerance:
        #    edge.next_node.edges[arg_max] = VEdge.zero()
        #else:
        #    edge.next_node.edges[arg_max].weight = np.complex128((mag_max / norm) + 0.j)

        # Actual normalization of the edges
        for i in range(len(edge.next_node.edges)):
        #if i != arg_max:
            edge.next_node.edges[i].weight = edge.next_node.edges[i].weight / common_factor #edge.weight
            if edge.next_node.edges[i].weight == 0. + 0.j:
                edge.next_node.edges[i] = VEdge.zero()

        return edge
    """

    @staticmethod
    def normalize(edge):
        zero = []
        non_zero_indices = []

        counter = 0
        for i in edge.next_node.edges:
            if np.abs(i.weight) < tolerance:
                zero.append(True)
            else:
                zero.append(False)
                non_zero_indices.append(counter)
            counter += 1

        if all(zero):
            return VEdge.zero

        if len(non_zero_indices) == 1:
            edge.weight = edge.next_node.edges[non_zero_indices[0]].weight
            edge.next_node.edges[non_zero_indices[0]].weight = 1.+0.j
            return edge

        sum_norm2 = np.abs(edge.next_node.edges[0].weight)
        mag2_max = np.abs(edge.next_node.edges[0].weight)
        arg_max = 0

        for i in range(1, len(edge.next_node.edges)):
            sum_norm2 += np.abs(edge.next_node.edges[i].weight)

        for i in range(1, len(edge.next_node.edges) + 1):
            counter_back = len(edge.next_node.edges) - i
            if np.abs(edge.next_node.edges[counter_back].weight) + tolerance >= mag2_max:
                mag2_max = np.abs(edge.next_node.edges[counter_back].weight)
                arg_max = counter_back

        norm = np.sqrt(sum_norm2)
        mag_max = np.sqrt(mag2_max)
        common_factor = norm / mag_max

        current_edge = edge.copy()
        max_edge = current_edge.next_node.edges[arg_max]

        real_part = current_edge.weight.real * common_factor
        img_part = current_edge.weight.imag * common_factor
        current_edge.weight = np.complex128(real_part+1j*img_part)
        if current_edge.weight == 0.+ 0.j:
            return VEdge.zero()

        max_edge.weight = np.complex128(mag_max / norm + 0.j)
        if max_edge.weight == 0.+0.j:
            current_edge.next_node.edges[arg_max] = VEdge.zero()

        for i in range(len(current_edge.next_node.edges)):
            if i != arg_max:
                # current_edge.next_node.edges[i]
                current_edge.next_node.edges[i].weight = current_edge.next_node.edges[i].weight / current_edge.weight
                if current_edge.next_node.edges[i].weight == 0.+0.j:
                    current_edge.next_node.edges[i] = VEdge.zero()

        return current_edge


class MEdgeNormalization:
    """
    @staticmethod
    def normalize(edge: MEdge) -> MEdge:
        arg_max = 0
        max_weight = edge.next_node.edges[0].weight

        for i in range(1, len(edge.next_node.edges) + 1):
            counter_back = len(edge.next_node.edges) - i
            if np.abs(edge.next_node.edges[counter_back].weight) + tolerance >= np.abs(max_weight):
                max_weight = edge.next_node.edges[counter_back].weight
                arg_max = counter_back

        if max_weight == 0.+0.j:
            return MEdge.zero()

        common_factor = max_weight

        current_edge = edge
        max_edge = current_edge.next_node.edges[arg_max]

        current_edge.weight = max_edge.weight

        if np.abs(current_edge.weight) < tolerance:
            return MEdge.zero().set_father_node(current_edge.next_node)

        # Actual normalization of the edges
        for i in range(len(edge.next_node.edges)):
            if i != arg_max:
                i_edge = edge.next_node.edges[i]
                if i_edge.weight != 0. + 0.j:  # Assuming 0.0 represents Complex::zero
                    edge.next_node.edges[i].weight = edge.next_node.edges[i].weight / common_factor
                else:
                    edge.next_node.edges[i] = MEdge.zero()

            else:
                edge.next_node.edges[i].weight = 1.+0.j

        return current_edge
    """
    @staticmethod
    def normalize(edge):
        argmax = -1

        zero = [i.weight == 0.+0.j for i in edge.next_node.edges]
        max_magnitude = 0
        max_weight = 1.+0.j

        for i in range(len(zero)):
            if zero[i]:
                continue

            if argmax == -1:
                argmax = i
                max_magnitude = np.abs(edge.next_node.edges[i].weight)
                max_weight = edge.next_node.edges[i].weight
            else:
                current_magnitude = np.abs(edge.next_node.edges[i].weight)
                if current_magnitude - max_magnitude > tolerance:
                    argmax = i
                    max_magnitude = current_magnitude
                    max_weight = edge.next_node.edges[i].weight

        if argmax == -1:
            return MEdge.zero()

        current_edge = edge.copy()

        for i in range(len(edge.next_node.edges)):
            if i == argmax:
                if current_edge.weight == 1.+0.j:
                    current_edge.weight = max_weight
                else:
                    current_edge.weight *= max_weight
                edge.next_node.edges[i].weight = 1.+0.j
            else:
                if not zero[i] and edge.next_node.edges[i].weight != 1.+0.j:
                    edge.next_node.edges[i].weight /= max_weight

        return current_edge

