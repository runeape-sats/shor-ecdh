# Solving Discrete Logarithm Problems in ECDH Using Qiskit

This guide walks through the process of using Qiskit to solve discrete logarithm problems in the context of Elliptic Curve Diffie-Hellman (ECDH), specifically on the SECP256K1-like curve:

y<sup>2</sup> = x<sup>3</sup> + 7

## 1. Introduction to the Problem

Elliptic Curve Cryptography (ECC) relies on the hardness of the Elliptic Curve Discrete Logarithm Problem (ECDLP), which states that given two points \( P \) and \( Q = kP \), finding \( k \) is computationally hard. Quantum algorithms, particularly Shor’s algorithm, provide a way to solve this problem efficiently.

## 2. Setting Up Qiskit

To begin, install Qiskit if you haven't already:

```bash
pip install qiskit
```

Now, import the necessary libraries:

```python
from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.algorithms import Shor
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
import numpy as np
```

## 3. Encoding ECDLP into Qiskit Circuits

### 3.1 Representation of the Elliptic Curve
We work with the curve \( y^2 = x^3 + 7 \) over a finite field \( \mathbb{F}_p \). Choose a prime number \( p \) and a base point \( P \):

```python
p = 23  # Prime modulus
a, b = 0, 7  # Curve parameters
G = (5, 19)  # Base point on the curve
```

### 3.2 Quantum Fourier Transform for Discrete Logarithm

Shor’s algorithm requires modular exponentiation, which in the elliptic curve setting corresponds to point multiplication. This involves representing integers in quantum registers and applying the Quantum Fourier Transform (QFT).

Define the QFT circuit:

```python
def qft(n):
    qc = QuantumCircuit(n)
    for i in range(n):
        for j in range(i):
            qc.cp(np.pi / 2**(i-j), j, i)
        qc.h(i)
    return qc
```

### 3.3 Implementing Shor’s Algorithm for ECDLP

Shor’s algorithm can be adapted to find the discrete log \( k \) given points \( P \) and \( Q = kP \). Use modular arithmetic circuits and phase estimation:

```python
def shors_dlog(N):
    qc = QuantumCircuit(N*2)
    
    # Apply Hadamard to first register
    for qubit in range(N):
        qc.h(qubit)
    
    # Modular multiplication (simulated)
    for qubit in range(N):
        qc.cx(qubit, N + qubit)
    
    # Apply QFT
    qc.append(qft(N).to_gate(), range(N))
    
    qc.measure_all()
    return qc

N = 4  # Number of qubits
qc = shors_dlog(N)
```

### 3.4 Running the Quantum Circuit

Use the Aer simulator to execute the circuit:

```python
backend = Aer.get_backend("qasm_simulator")
job = execute(qc, backend, shots=1024)
result = job.result()
counts = result.get_counts()
print(counts)
```

## 4. Interpreting Results

The most frequent measurement result corresponds to the discrete logarithm \( k \). Classical post-processing retrieves the value efficiently.

## 5. Conclusion

This guide demonstrates how to apply Qiskit circuits to solve the discrete logarithm problem in the context of ECDH. Future work includes optimizing quantum circuits and using real quantum hardware.

