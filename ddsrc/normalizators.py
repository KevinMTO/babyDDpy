import numpy as np

from ddsrc.medge import MEdge
from ddsrc.vedge import VEdge

tolerance = 1e-16


class VEdgeNormalization:
    @staticmethod
    def normalize(edge: VEdge) -> VEdge:
        sum_norm2 = np.abs(edge.next_node.edges[0].weight) ** 2
        mag2_max = np.abs(edge.next_node.edges[0].weight) ** 2
        arg_max = 0

        for i in range(1, len(edge.next_node.edges)):
            sum_norm2 += np.abs(edge.next_node.edges[i].weight) ** 2

        for i in range(1, len(edge.next_node.edges) + 1):
            counter_back = len(edge.next_node.edges) - i
            if np.abs(edge.next_node.edges[counter_back].weight) + tolerance >= mag2_max:
                mag2_max = np.abs(edge.next_node.edges[counter_back].weight) ** 2
                arg_max = counter_back

        norm = sum_norm2 ** 0.5
        mag_max = mag2_max ** 0.5
        if norm == 0.0:
            return MEdge.zero()
        else:
            common_factor = norm / mag_max

        current_edge = edge
        max_edge = current_edge.next_node.edges[arg_max]

        if max_edge.weight != 1.0:
            current_edge.weight = max_edge.weight
            real_part = np.real(current_edge.weight) * common_factor
            img_part = np.imag(current_edge.weight) * common_factor
            current_edge.weight = np.complex128(real_part + 1j * img_part)
        else:
            real_part = np.real(current_edge.weight) * common_factor
            img_part = np.imag(current_edge.weight) * common_factor
            current_edge.weight = np.complex128(real_part + 1j * img_part)

            if np.abs(current_edge.weight) < tolerance:
                return MEdge.zero()

        max_edge.weight = np.complex128(mag_max / norm)
        if np.abs(max_edge.weight) < tolerance:
            current_edge.next_node.edges[arg_max] = MEdge.zero().set_father_node(current_edge.next_node)

        # Actual normalization of the edges
        for i in range(len(edge.next_node.edges)):
            if i != arg_max:
                i_edge = edge.next_node.edges[i]

                if i_edge.weight != 0. + 0.j:  # Assuming 0.0 represents Complex::zero
                    if (np.real(i_edge.weight) == 0.):
                        real_part = 0.
                    else:
                        real_part = np.real(i_edge.weight) / np.real(current_edge.weight)
                    if (np.imag(i_edge.weight) == 0.):
                        img_part = 0.
                    else:
                        img_part = np.imag(i_edge.weight) / np.imag(current_edge.weight)
                    i_edge.weight = np.complex128(real_part + 1j * img_part)
                else:
                    edge.next_node.edges[i] = MEdge.zero()

        return current_edge


class MEdgeNormalization:

    @staticmethod
    def normalize(edge: MEdge) -> MEdge:
        sum_norm2 = np.abs(edge.next_node.edges[0].weight) ** 2
        mag2_max = np.abs(edge.next_node.edges[0].weight) ** 2
        arg_max = 0

        for i in range(1, len(edge.next_node.edges)):
            sum_norm2 += np.abs(edge.next_node.edges[i].weight) ** 2

        for i in range(1, len(edge.next_node.edges) + 1):
            counter_back = len(edge.next_node.edges) - i
            if np.abs(edge.next_node.edges[counter_back].weight) + tolerance >= mag2_max:
                mag2_max = np.abs(edge.next_node.edges[counter_back].weight) ** 2
                arg_max = counter_back

        norm = sum_norm2 ** 0.5
        mag_max = mag2_max ** 0.5
        if norm == 0.0:
            return MEdge.zero()
        else:
            common_factor = norm / mag_max

        current_edge = edge
        max_edge = current_edge.next_node.edges[arg_max]

        if max_edge.weight != 1.0:
            current_edge.weight = max_edge.weight
            real_part = np.real(current_edge.weight) * common_factor
            img_part = np.imag(current_edge.weight) * common_factor
            current_edge.weight = np.complex128(real_part + 1j * img_part)
        else:
            real_part = np.real(current_edge.weight) * common_factor
            img_part = np.imag(current_edge.weight) * common_factor
            current_edge.weight = np.complex128(real_part + 1j * img_part)

            if np.abs(current_edge.weight) < tolerance:
                return MEdge.zero()

        max_edge.weight = np.complex128(mag_max / norm)
        if np.abs(max_edge.weight) < tolerance:
            current_edge.next_node.edges[arg_max] = MEdge.zero().set_father_node(current_edge.next_node)

        # Actual normalization of the edges
        for i in range(len(edge.next_node.edges)):
            if i != arg_max:
                i_edge = edge.next_node.edges[i]

                if i_edge.weight != 0. + 0.j:  # Assuming 0.0 represents Complex::zero
                    if (np.real(i_edge.weight) == 0.):
                        real_part = 0.
                    else:
                        real_part = np.real(i_edge.weight) / np.real(current_edge.weight)
                    if (np.imag(i_edge.weight) == 0.):
                        img_part = 0.
                    else:
                        img_part = np.imag(i_edge.weight) / np.imag(current_edge.weight)
                    i_edge.weight = np.complex128(real_part + 1j * img_part)
                else:
                    edge.next_node.edges[i] = MEdge.zero()

        return current_edge
