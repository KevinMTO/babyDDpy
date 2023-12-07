from itertools import product

from ddsrc.edge import Edge, one
from ddsrc.medge import MEdge
from ddsrc.normalizators import MEdgeNormalization, VEdgeNormalization
from ddsrc.tables import ComputeTable, UniqueTable
from ddsrc.vedge import VEdge
import numpy as np

NDArray = np.ndarray


class DDPackage:
    def __init__(self, labels):
        self.unique_table = UniqueTable()
        self.compute_table = ComputeTable()
        self.registersSizes = len(labels) * [2]
        self.numberOfQuantumRegisters = len(labels)
        self.labels = labels

    def normalize(self, new_edge) -> Edge:
        if isinstance(new_edge, VEdge):
            normalizer = VEdgeNormalization()
        else:
            normalizer = MEdgeNormalization()

        return normalizer.normalize(new_edge)

    def makeZeroState(self, n):
        if n > self.numberOfQuantumRegisters:
            raise RuntimeError("Requested state with {} QUANTUM REGISTERS, "
                               "but the current package configuration only "
                               "supports up to {} QUANTUM REGISTERS. Please "
                               "allocate a larger package instance.".format(n, self.numberOfQuantumRegisters))

        first = VEdge.one()
        for nodeIdx in range(0, n):
            new_outgoing_edges = [first] + [VEdge.zero() for _ in range(self.registersSizes[nodeIdx] - 1)]
            first = self.makeDDNode(nodeIdx, new_outgoing_edges)

        return first

    def makeDDNode(self, idx, edges):
        unique_table = self.unique_table

        if isinstance(edges[0], VEdge):
            Edge = VEdge
        else:
            Edge = MEdge

        new_edge = Edge(1. + 0.j, unique_table.get_node())
        new_edge.next_node.index = idx
        new_edge.next_node.edges = edges

        for e in edges:
            e.father_node = new_edge.next_node

        for edge in edges:
            assert edge.next_node.index == idx - 1 or edge.is_terminal()

        # Placeholder for normalize function
        new_edge = self.normalize(new_edge)
        assert new_edge.next_node.index == idx or new_edge.is_terminal()

        looked_up_edge = unique_table.lookup(new_edge)
        assert looked_up_edge.next_node.index == idx or looked_up_edge.is_terminal()

        return looked_up_edge

    def makeGateDD(self, mat, n, controls, target) -> MEdge:
        if n > self.numberOfQuantumRegisters:
            raise RuntimeError(
                    f"Requested gate with {n} qubits, but current package configuration "
                    f"only supports up to {self.numberOfQuantumRegisters} qubits."
                    f" Please allocate a larger package instance.")

        target_radix = self.registersSizes[target]

        edges = target_radix * target_radix
        edges_mat = [MEdge.zero() for _ in range(edges)]

        current_control = iter(controls)
        current_control = next(current_control, None)

        flat_mat = mat.flatten()
        for i in range(edges):
            if flat_mat[i].real != 0 or flat_mat[i].imag != 0:
                new_terminal_edge = MEdge(flat_mat[i], one)
                edges_mat[i] = new_terminal_edge

        current_reg = 0

        # process lines below target
        while current_reg < target:
            radix = self.registersSizes[current_reg]

            for row_mat in range(target_radix):
                for col_mat in range(target_radix):

                    entry_pos = (row_mat * target_radix) + col_mat
                    quad_edges = [MEdge.zero() for _ in range(radix * radix)]

                    if current_control and current_control.index == current_reg:  # ????????
                        if row_mat == col_mat:
                            for i in range(radix):
                                diag_ind = i * radix + i

                                if i == current_control.level:
                                    quad_edges[diag_ind] = edges_mat[entry_pos]
                                else:
                                    quad_edges[diag_ind] = self.makeIdent(0,
                                                                          current_reg - 1)  # fixed to start from zero
                        else:
                            quad_edges[current_control.level + radix * current_control.level] = edges_mat[
                                entry_pos]

                        edges_mat[entry_pos] = self.makeDDNode(current_reg, quad_edges)

                    else:  # not connected
                        for i_d in range(radix):
                            quad_edges[i_d * radix + i_d] = edges_mat[entry_pos]

                        edges_mat[entry_pos] = self.makeDDNode(current_reg, quad_edges)

            if current_control and current_control.index == current_reg:
                current_control = next(current_control, None)

            current_reg += 1

        # target line
        target_node_edge = self.makeDDNode(current_reg, edges_mat)

        # process lines above target
        while current_reg < n - 1:  # fixed to start from zero
            next_reg = current_reg + 1
            next_radix = self.registersSizes[next_reg]
            next_edges = [MEdge.zero() for _ in range(next_radix * next_radix)]

            if current_control and current_control.index == next_reg:
                for i in range(next_radix):
                    diag_ind = i * next_radix + i

                    if i == current_control.type:
                        next_edges[diag_ind] = target_node_edge
                    else:
                        next_edges[diag_ind] = self.makeIdent(0, next_reg - 1)

                current_control = next(current_control, None)

            else:  # not connected
                for i_d in range(next_radix):
                    next_edges[i_d * next_radix + i_d] = target_node_edge

            target_node_edge = self.makeDDNode(next_reg, next_edges)

            current_reg += 1

        return target_node_edge

    def makeIdent(self, least_significant_qubit, most_significant_qubit) -> MEdge:
        if most_significant_qubit < least_significant_qubit:
            return MEdge.one()

        if least_significant_qubit == 0 and self.idTable[most_significant_qubit].nextNode is not None:
            return self.idTable[most_significant_qubit]

        if most_significant_qubit >= 1 and self.idTable[most_significant_qubit - 1].nextNode is not None:
            basic_dim_most = self.registersSizes[most_significant_qubit]
            identity_edges = []

            for i in range(basic_dim_most):
                for j in range(basic_dim_most):
                    if i == j:
                        identity_edges.append(self.idTable[most_significant_qubit - 1])
                    else:
                        identity_edges.append(mEdge.zero())

            self.idTable[most_significant_qubit] = makeDDNode(most_significant_qubit, identity_edges)
            return self.idTable[most_significant_qubit]

        basic_dim_least = self.registersSizes[least_significant_qubit]
        identity_edges_least = []

        for i in range(basic_dim_least):
            for j in range(basic_dim_least):
                if i == j:
                    identity_edges_least.append(mEdge.one())
                else:
                    identity_edges_least.append(mEdge.zero())

        e = self.makeDDNode(least_significant_qubit, identity_edges_least)

        for intermediary_regs in range(least_significant_qubit + 1, most_significant_qubit + 1):
            basic_dim_int = self.registersSizes[intermediary_regs]
            identity_edges_int = []

            for i in range(basic_dim_int):
                for j in range(basic_dim_int):
                    if i == j:
                        identity_edges_int.append(e)
                    else:
                        identity_edges_int.append(mEdge.zero())

            e = self.makeDDNode(intermediary_regs, identity_edges_int)

        if least_significant_qubit == 0:
            self.idTable[most_significant_qubit] = e

        return e

    def display_path(self, state: VEdge, bitstring: str):
        print(state.weight)

        # Base case: if the path is empty, return the node's weight
        if state.is_terminal():
            return

        # Get the next edge index based on the path
        next_edge_index = int(bitstring[0])

        # Check if the index is valid
        if 0 <= next_edge_index < len(state.next_node.edges):
            # Recursively call the function with the next node and the remaining path
            next_edge = state.next_node.edges[next_edge_index]
            remaining_path = bitstring[1:]
            return self.display_path(next_edge, remaining_path)

    def create_zero_state(self, num_qubits):
        return self.makeZeroState(num_qubits)

    def create_single_qubit_gate(self, num_qubits: int, target: int, gate_matrix: NDArray[np.complex128]):
        return self.makeGateDD(gate_matrix, num_qubits, [], target)

    # def create_nearest_neighbor_two_qubit_gate(self, num_qubits: int, qubit0: int, qubit1: int,
    #                                           gate_matrix: NDArray[np.complex128]):
    #    pass

    def create_controlled_single_qubit_gate(self, num_qubits: int, control: int, target: int,
                                            gate_matrix: NDArray[np.complex128]):
        return self.makeGateDD(gate_matrix, num_qubits, control, target)

    def apply_gate(self, state: VEdge, gate: MEdge):
        pass

    def get_amplitude(self, state: VEdge, bitstring: str) -> np.complex128:
        # Base case: if the path is empty, return the node's weight
        if state.is_terminal():
            return state.weight

        # Get the next edge index based on the path
        next_edge_index = int(bitstring[0])

        # Check if the index is valid
        if 0 <= next_edge_index < len(state.next_node.edges):
            # Recursively call the function with the next node and the remaining path
            next_edge = state.next_node.edges[next_edge_index]
            remaining_path = bitstring[1:]
            return state.weight * self.get_amplitude(next_edge, remaining_path)
        else:
            # Invalid edge index, return 0
            raise IndexError

    def measure_all(self, state: VEdge, shots: int) -> dict[str, int]:
        res = {}
        binary_strings = [format(i, '0{}b'.format(len(self.registersSizes))) for i in
                          range(2 ** len(self.registersSizes))]
        for bitstring in binary_strings:
            res[bitstring] = self.get_amplitude(state, bitstring)
        return res

    def gate_print(self, mat):
        res={}
        binary_strings = ['00', '01', '02', '03', '10', '11', '12', '13', '20', '21', '22', '23', '30', '31', '32', '33']

        print(binary_strings)
        for bitstring in binary_strings:
            res[bitstring] = self.get_amplitude(mat, bitstring)
        return res

    def simulate_ghz_state(self, num_qubits: int, shots: int) -> dict[str, int]:
        pass
