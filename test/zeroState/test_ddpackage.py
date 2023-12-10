from unittest import TestCase

import numpy as np
from ddsrc.ddpackage import DDPackage


class TestDDPackage(TestCase):
    def test_create_zero_state(self):
        num_qubits = 5
        dd = DDPackage(num_qubits)
        zero = dd.create_zero_state(num_qubits)
        print(dd.measure_all(zero))
        #print("------------")
        #dd.display_path(zero, "00")
        #print("------------")
        #print(dd.get_amplitude(zero, "00"))
        #print("------------")

        x = dd.create_single_qubit_gate(num_qubits, 0, np.array([[0, 1], [1, 0]], dtype=np.complex128))
        hadamard = dd.create_single_qubit_gate(num_qubits, 1, 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]],
                                                                                        dtype=np.complex128))
        #print(dd.gate_print_two_qubit_debug(x))
        cx = dd.create_controlled_single_qubit_gate(num_qubits, [1], 0, np.array([[0, 1], [1, 0]]))

        psi = dd.apply_gate(hadamard, zero)
        psi = dd.apply_gate(hadamard, psi)
        psi = dd.apply_gate(cx, psi)
        print(dd.measure_all(psi))


