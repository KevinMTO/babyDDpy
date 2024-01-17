# Quantum Decision Diagram Python Package

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/downloads/release)


## Overview

This Python package provides a powerful and flexible implementation of decision diagrams for efficient representation and manipulation of quantum computations. Decision diagrams are commonly used in symbolic model checking, formal verification, simulation and optimization problems.

The library supports the usage of qudits and qubits, a.k.a. mixed-dimensional systems.

```
num_qubits = 5
dd = DDPackage(num_qubits)
zero = dd.create_zero_state(num_qubits)
print(dd.measure_all(zero))

hadamard = dd.create_single_qubit_gate(num_qubits, 1, 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]],
                                                                                dtype=np.complex128))

cx = dd.create_controlled_single_qubit_gate(num_qubits, [1], 0, np.array([[0, 1], [1, 0]]))

psi = dd.apply_gate(hadamard, zero)
psi = dd.apply_gate(hadamard, psi)
psi = dd.apply_gate(cx, psi)
print(dd.measure_all(psi))
```
# Contributing

We welcome contributions! If you would like to contribute to this project, please open a PR.

# License

This project is licensed under the MIT License.

# Acknoledgemnts
The repo is inspired by :

K. Mato, S. Hillmich and R. Wille, "Mixed-Dimensional Quantum Circuit Simulation with Decision Diagrams," 2023 IEEE International Conference on Quantum Computing and Engineering (QCE), Bellevue, Washington, USA, 2023.
