import numpy as np

from ddsrc.edge_nodes import MEdge, VEdge

tolerance = 1e-16


class VEdgeNormalization:

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
            edge.next_node.edges[non_zero_indices[0]].weight = 1. + 0.j
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
        current_edge.weight = np.complex128(real_part + 1j * img_part)
        if current_edge.weight == 0. + 0.j:
            return VEdge.zero()

        max_edge.weight = np.complex128(mag_max / norm + 0.j)
        if max_edge.weight == 0. + 0.j:
            current_edge.next_node.edges[arg_max] = VEdge.zero()

        for i in range(len(current_edge.next_node.edges)):
            if i != arg_max:
                # current_edge.next_node.edges[i]
                current_edge.next_node.edges[i].weight = current_edge.next_node.edges[i].weight / current_edge.weight
                if current_edge.next_node.edges[i].weight == 0. + 0.j:
                    current_edge.next_node.edges[i] = VEdge.zero()

        return current_edge


class MEdgeNormalization:

    @staticmethod
    def normalize(edge):
        argmax = -1

        zero = [i.weight == 0. + 0.j for i in edge.next_node.edges]
        max_magnitude = 0
        max_weight = 1. + 0.j

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
                if current_edge.weight == 1. + 0.j:
                    current_edge.weight = max_weight
                else:
                    current_edge.weight *= max_weight
                edge.next_node.edges[i].weight = 1. + 0.j
            else:
                if not zero[i] and edge.next_node.edges[i].weight != 1. + 0.j:
                    edge.next_node.edges[i].weight /= max_weight

        return current_edge
