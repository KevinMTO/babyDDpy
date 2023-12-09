from functools import reduce

from numpy._typing import _64Bit
from ddsrc.control import Control
from ddsrc.edge_nodes import Edge, MEdge, VEdge, one
from ddsrc.normalizators import MEdgeNormalization, VEdgeNormalization, tolerance
from ddsrc.tables import AdditionTable, MultTable, UniqueTable
import numpy as np

NDArray = np.ndarray


class DDPackage:
    def __init__(self, num_lines):
        self.id_table = {}
        for i in range(num_lines):
            self.id_table[i] = MEdge(None, None)
        self.unique_table = UniqueTable()
        self.add_table = AdditionTable()
        self.mul_table = MultTable()
        self.registers_sizes = num_lines * [2]
        self.numberOfQuantumRegisters = num_lines

    def multiply(self, x, y):

        var = -1

        if not x.is_terminal():
            var = x.next_node.index
        if not y.is_terminal() and y.next_node.index > var:
            var = y.next_node.index

        e = self.multiply2(x, y, var)

        return e

    def multiply2(self, x, y, var):
        LEdge = type(x)
        REdge = type(y)
        ResultEdge = type(y)

        if x.next_node is None:
            return ResultEdge.zero()
        if y.next_node is None:
            return y

        if x.weight == 0. + 0.j or y.weight == 0. + 0.j:
            return ResultEdge.zero()

        if var == -1:
            return ResultEdge(x.weight * y.weight, one)

        # x_copy = x.copy()
        # y_copy = y.copy()

        compute_table = self.mul_table
        lookup_result = compute_table.lookup(x, y)  # x_copy, y_copy)

        if lookup_result.next_node is not None:
            if np.abs(lookup_result.weight) < tolerance:
                return ResultEdge.zero()

            res_edge_init = lookup_result.copy()
            res_edge_init.weight = res_edge_init.weight * x.weight * y.weight

            if np.abs(res_edge_init.weight) < tolerance:
                return ResultEdge.zero()

            return res_edge_init

        rows = 1 if x.is_terminal() else self.registers_sizes[x.next_node.index]
        # cols = 1 if isinstance(y, MEdge) else self.registers_sizes[y.next_node.index] if not y.is_terminal() else 1
        cols = 1 if isinstance(REdge, MEdge) else (
            1 if not y.is_terminal() else self.registers_sizes[int(y.next_node.index)])

        multiplication_boundary = rows

        edge = [ResultEdge.zero() for _ in range(multiplication_boundary * cols)]

        for i in range(rows):
            for j in range(cols):
                idx = cols * i + j

                for k in range(multiplication_boundary):
                    e1 = LEdge() if x.is_terminal() or x.next_node.index != var else x.next_node.edges[rows * i + k]
                    e2 = REdge() if y.is_terminal() or y.next_node.index != var else y.next_node.edges[j + cols * k]

                    multiplied_recur_res = self.multiply2(e1, e2, var - 1)

                    if k == 0 or edge[idx].weight == 0. + 0.j:
                        edge[idx] = multiplied_recur_res
                    elif multiplied_recur_res.weight != 0. + 0.j:
                        old_edge = edge[idx].copy()
                        edge[idx] = self.add(old_edge, multiplied_recur_res)

        result_edge = self.makeDDNode(var, edge)

        compute_table.insert(x, y, Edge(next_node=result_edge.next_node,
                                        weight=result_edge.weight))  # x_copy, y_copy, Edge(next_node=result_edge.next_node, weight=result_edge.weight))

        if result_edge.weight != 0. + 0.j and (x.weight != 1. + 0.j or y.weight != 1. + 0.j):
            if result_edge.weight == 1. + 0.j:
                result_edge.weight = x.weight * y.weight
            else:
                result_edge.weight = result_edge.weight * x.weight
                result_edge.weight = result_edge.weight * y.weight

            if np.abs(result_edge.weight) < tolerance:
                return Edge.zero()

        return result_edge

    def add(self, x, y):
        if isinstance(x, VEdge) and isinstance(y, VEdge):
            Edge = VEdge
        elif isinstance(x, MEdge) and isinstance(y, MEdge):
            Edge = MEdge
        else:
            raise TypeError("Edge type not matching")

        if x.next_node is None:
            return y
        if y.next_node is None:
            return x

        if x.weight == 0. + 0.j:
            if y.weight == 0. + 0.j:
                return y
            result = y.copy()
            result.weight = x.weight
            return result

        if y.weight == 0. + 0.j:
            result = x.copy()
            result.weight = y.weight
            return result

        if x.next_node == y.next_node:
            result = y.copy()
            result.weight = x.weight + y.weight
            if np.abs(result.weight) < tolerance:
                return Edge.zero()

            return result

        compute_table = self.add_table
        result = compute_table.lookup(x, y)

        if result.next_node is not None:
            if np.abs(result.weight) < tolerance:
                return Edge.zero()

            return result.copy()

        ##################################################################################################
        ##################################################################################################

        if x.is_terminal():
            new_successor = y.next_node.index
        else:
            new_successor = x.next_node.index
            if not y.is_terminal() and y.next_node.index > new_successor:
                new_successor = y.next_node.index

        edge_sum = [Edge.zero() for _ in range(len(x.next_node.edges))]

        for i in range(len(x.next_node.edges)):

            if not x.is_terminal() and x.next_node.index == new_successor:
                e1 = x.next_node.edges[i]

                if e1.weight != 0. + 0.j:
                    e1.weight = e1.weight * x.weight
            else:
                e1 = x.copy()
                if y.next_node.edges[i].next_node is None:
                    e1 = Edge(0. + 0.j, None)

            if not y.is_terminal() and y.next_node.index == new_successor:
                e2 = y.next_node.edges[i]

                if e2.weight != 0. + 0.j:
                    e2.weight = e2.weight * y.weight
            else:
                e2 = y.copy()
                if x.next_node.edges[i].next_node is None:
                    e2 = Edge(0. + 0.j, None)

            edge_sum[i] = self.add(e1, e2)

        e = self.makeDDNode(new_successor, edge_sum)
        compute_table.insert(x, y, e)
        return e

    #############################################################################################################
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
            new_outgoing_edges = [first] + [VEdge.zero() for _ in range(self.registers_sizes[nodeIdx] - 1)]
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
        #new_edge = self.normalize(new_edge)
        assert new_edge.next_node.index == idx or new_edge.is_terminal()

        looked_up_edge = unique_table.lookup(new_edge)
        assert looked_up_edge.next_node.index == idx or looked_up_edge.is_terminal()

        return looked_up_edge

    def makeGateDD(self, mat, n, controls, target) -> MEdge:
        if not isinstance(controls, list):
            raise TypeError

        if n > self.numberOfQuantumRegisters:
            raise RuntimeError(
                    f"Requested gate with {n} qubits, but current package configuration "
                    f"only supports up to {self.numberOfQuantumRegisters} qubits."
                    f" Please allocate a larger package instance.")

        target_radix = self.registers_sizes[target]

        edges = target_radix * target_radix
        edges_mat = [MEdge.zero() for _ in range(edges)]

        counter_controls = 0
        try:
            current_control = controls[counter_controls]
        except IndexError:
            current_control = None

        flat_mat = mat.flatten()
        for i in range(edges):
            if flat_mat[i].real != 0 or flat_mat[i].imag != 0:
                new_terminal_edge = MEdge(flat_mat[i], one)
                edges_mat[i] = new_terminal_edge

        current_reg = 0

        # process lines below target
        while current_reg < target:
            radix = self.registers_sizes[current_reg]

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
                counter_controls += 1
                try:
                    current_control = controls[counter_controls]
                except IndexError:
                    current_control = None

            current_reg += 1

        # target line
        target_node_edge = self.makeDDNode(current_reg, edges_mat)

        # process lines above target
        while current_reg < n - 1:  # fixed to start from zero
            next_reg = current_reg + 1
            next_radix = self.registers_sizes[next_reg]
            next_edges = [MEdge.zero() for _ in range(next_radix * next_radix)]

            if current_control and current_control.index == next_reg:
                for i in range(next_radix):
                    diag_ind = i * next_radix + i

                    if i == current_control.level:
                        next_edges[diag_ind] = target_node_edge
                    else:
                        next_edges[diag_ind] = self.makeIdent(0, next_reg - 1)

                counter_controls += 1
                try:
                    current_control = controls[counter_controls]
                except IndexError:
                    current_control = None

            else:  # not connected
                for i_d in range(next_radix):
                    next_edges[i_d * next_radix + i_d] = target_node_edge

            target_node_edge = self.makeDDNode(next_reg, next_edges)

            current_reg += 1

        return target_node_edge

    def makeIdent(self, least_significant_qubit, most_significant_qubit) -> MEdge:
        if most_significant_qubit < least_significant_qubit:
            return MEdge.one()

        if least_significant_qubit == 0 and self.id_table[most_significant_qubit].next_node is not None:
            return self.id_table[most_significant_qubit]

        if most_significant_qubit >= 1 and self.id_table[most_significant_qubit - 1].next_node is not None:
            basic_dim_most = self.registers_sizes[most_significant_qubit]
            identity_edges = []

            for i in range(basic_dim_most):
                for j in range(basic_dim_most):
                    if i == j:
                        identity_edges.append(self.id_table[most_significant_qubit - 1])
                    else:
                        identity_edges.append(MEdge.zero())

            self.id_table[most_significant_qubit] = self.makeDDNode(most_significant_qubit, identity_edges)
            return self.id_table[most_significant_qubit]

        basic_dim_least = self.registers_sizes[least_significant_qubit]
        identity_edges_least = []

        for i in range(basic_dim_least):
            for j in range(basic_dim_least):
                if i == j:
                    identity_edges_least.append(MEdge.one())
                else:
                    identity_edges_least.append(MEdge.zero())

        e = self.makeDDNode(least_significant_qubit, identity_edges_least)

        for intermediary_regs in range(least_significant_qubit + 1, most_significant_qubit + 1):
            basic_dim_int = self.registers_sizes[intermediary_regs]
            identity_edges_int = []

            for i in range(basic_dim_int):
                for j in range(basic_dim_int):
                    if i == j:
                        identity_edges_int.append(e)
                    else:
                        identity_edges_int.append(MEdge.zero())

            e = self.makeDDNode(intermediary_regs, identity_edges_int)

        if least_significant_qubit == 0:
            self.id_table[most_significant_qubit] = e

        return e

    ############################################################################################################
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

    def collect_path(self, state: VEdge, bitstring: str, collector):
        collector.append(state.weight)

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
            return self.collect_path(next_edge, remaining_path, collector)

    def gate_print_two_qubit_debug(self, mat):
        res = {}
        binary_strings = ['00', '01', '02', '03', '10', '11', '12', '13', '20', '21', '22', '23', '30', '31', '32',
                          '33']
        for bitstring in binary_strings:
            res[bitstring] = self.get_amplitude(mat, bitstring)
        return res

    #########################################################################################################

    def create_zero_state(self, num_qubits):
        return self.makeZeroState(num_qubits)

    def create_single_qubit_gate(self, num_qubits: int, target: int, gate_matrix: NDArray[np.complex128]):
        return self.makeGateDD(gate_matrix, num_qubits, [], target)

    def create_controlled_single_qubit_gate(self, num_qubits: int, controls: list[int],
                                            target: int, gate_matrix: NDArray[np.complex128]):
        controls = [Control(cq, 1) for cq in controls]
        return self.makeGateDD(gate_matrix, num_qubits, controls, target)

    def apply_gate(self, gate: MEdge, state: VEdge) -> VEdge:
        return self.multiply(gate, state)

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
            # Invalid edge index
            raise IndexError

    def measure_all(self, state: VEdge) -> dict[str, np.complexfloating[_64Bit, _64Bit]]:
        res = {}
        binary_strings = [format(i, '0{}b'.format(len(self.registers_sizes))) for i in
                          range(2 ** len(self.registers_sizes))]
        for bitstring in binary_strings:
            res[bitstring] = self.get_amplitude(state, bitstring)
        return res

    def measure_all_by_reduction(self, state: VEdge):
        res = {}
        binary_strings = [format(i, '0{}b'.format(len(self.registers_sizes))) for i in
                          range(2 ** len(self.registers_sizes))]
        for bitstring in binary_strings:
            nums = []
            self.collect_path(state, bitstring, nums)
            res[bitstring] = reduce(lambda x, y: x * np.abs(y), nums, 1)
        return res

    def simulate_ghz_state(self, num_qubits: int, shots: int) -> dict[str, int]:
        pass
