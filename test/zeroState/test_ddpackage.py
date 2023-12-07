from unittest import TestCase

import numpy as np

from ddsrc.ddpackage import DDPackage


class TestDDPackage(TestCase):
    def test_create_zero_state(self):
        dd = DDPackage(["q0", "q1"])
        zero = dd.create_zero_state(2)
        print(dd.measure_all(zero, 1000))
        print("------------")
        dd.display_path(zero, "00")
        print("------------")
        print(dd.get_amplitude(zero, "00"))
        print("------------")
        gate = dd.create_single_qubit_gate(2, 1, np.array([[0, 1], [1, 0]], dtype=np.complex128))
        print(dd.gate_print(gate))
        x = 0
