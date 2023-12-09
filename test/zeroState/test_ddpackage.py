from unittest import TestCase

import numpy as np

from ddsrc.control import Control
from ddsrc.ddpackage import DDPackage


class TestDDPackage(TestCase):
    def test_create_zero_state(self):
        dd = DDPackage(2)
        zero = dd.create_zero_state(2)
        print(dd.measure_all(zero))
        #print("------------")
        #dd.display_path(zero, "00")
        #print("------------")
        #print(dd.get_amplitude(zero, "00"))
        #print("------------")
        x = dd.create_single_qubit_gate(2, 0, np.array([[0, 1], [1, 0]], dtype=np.complex128))
        hadamard = dd.create_single_qubit_gate(2, 1, 1/np.sqrt(2)*np.array([[1, 1], [1, -1]], dtype=np.complex128))
        #print(dd.gate_print_two_qubit_debug(x))
        #gate2 = dd.create_single_qubit_gate(2, 1, np.array([[0, 1], [1, 0]], dtype=np.complex128))
        cx = dd.create_controlled_single_qubit_gate(2, [1], 0, np.array([[0, 1], [1, 0]]))
        #print(dd.gate_print_two_qubit_debug(gate2))
        #print(dd.measure_all(dd.add(zero, zero)))
        psi = dd.apply_gate(hadamard, zero)
        #dd.display_path(psi, "00")
        print(dd.measure_all(psi))
        #print(dd.get_amplitude(psi, "00"))
        psi = dd.apply_gate(cx, psi)
        #dd.display_path(psi, "00")
        print(dd.measure_all(psi))

        #print(dd.measure_all_2(psi))


        x = 0
