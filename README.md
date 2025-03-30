# Solving Discrete Logarithm Problems in ECDH Using Qiskit

v0.0.1

This guide demonstrates how to use Qiskit to explore the Elliptic Curve Discrete Logarithm Problem (ECDLP) in the context of Elliptic Curve Diffie-Hellman (ECDH), focusing on the curve:

$$ y^2 = x^3 + 7 $$

We’ll use a simplified example to illustrate the quantum approach, adapting Shor’s algorithm conceptually for ECDLP.

## 1. Introduction to the Problem

Elliptic Curve Cryptography (ECC) underpins ECDH and relies on the computational hardness of the ECDLP: given points $P$ (a generator) and $Q = kP$ on an elliptic curve, finding the integer $k$ is infeasible with classical computers. Quantum computing, particularly Shor’s algorithm, offers a polynomial-time solution, threatening ECC’s security. Here, we simulate this process using Qiskit on a small-scale curve resembling SECP256K1’s form.

## 2. Setting Up Qiskit

Install Qiskit if you haven’t already:

```bash
pip install qiskit qiskit-aer
```

Import the required libraries:

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np
```

## 3. Encoding ECDLP into Qiskit Circuits

### 3.1 Representation of the Elliptic Curve

We define the curve $y^2 = x^3 + 7$ over a finite field $\mathbb{F}_p$, where $p$ is a small prime for demonstration (real-world curves like SECP256K1 use much larger primes, e.g., $p \approx 2^{256}$):

```python
p = 17  # Small prime for simplicity
a, b = 0, 7  # Curve: y^2 = x^3 + 7
G = (5, 8)  # Base point (verified: 8^2 ≡ 5^3 + 7 ≡ 13 mod 17)
```

Verify $G = (5, 8)$ lies on the curve:
- Left: $8^2 = 64 \equiv 13 \pmod{17}$
- Right: $5^3 + 7 = 125 + 7 = 132 \equiv 13 \pmod{17}$
- $13 = 13$, so $(5, 8)$ is valid.

### 3.2 Quantum Fourier Transform for Discrete Logarithm

Shor’s algorithm uses the Quantum Fourier Transform (QFT) to extract periodicity. For ECDLP, we seek $k$ where $Q = kP$, leveraging QFT to estimate phases related to the curve’s order. Define the QFT circuit:

```python
def qft(n):
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, n):
            qc.cp(np.pi / 2**(j - i), j, i)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc
```

### 3.3 Implementing Shor’s Algorithm for ECDLP

Shor’s algorithm for classic discrete logs ($g^k \equiv h \pmod{p}$) uses modular exponentiation. For ECDLP, we need elliptic curve point multiplication ($kP$), which is complex to implement quantumly. Here, we simulate the structure with a simplified circuit:

```python
def shors_dlog(n_qubits):
    qc = QuantumCircuit(2 * n_qubits, n_qubits)  # Two registers: control and target
    # Apply Hadamard gates to control register
    for qubit in range(n_qubits):
        qc.h(qubit)
    # Placeholder for EC point multiplication (Q = kP)
    # In practice, this requires custom gates for elliptic curve arithmetic
    for qubit in range(n_qubits):
        qc.cx(qubit, n_qubits + qubit)  # Simplified stand-in for demonstration
    # Apply QFT to control register
    qc.append(qft(n_qubits).to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc

n = 4  # Small qubit count for simulation
qc = shors_dlog(n)
```

**Note**: This is a conceptual simplification. Real ECDLP solvers need:
- Quantum circuits for elliptic curve addition/multiplication.
- The curve’s order $n$ (where $nP = \mathcal{O}$, the point at infinity).

### 3.4 Running the Quantum Circuit

Execute the circuit using the AerSimulator:

```python
backend = AerSimulator()
t_qc = qc.decompose()  # Optional: visualize gate decomposition
result = backend.run(t_qc, shots=1024).result()
counts = result.get_counts()
print("Measurement results:", counts)
```

## 4. Interpreting Results

The output provides phase estimates tied to $k$ modulo the curve’s order. In Shor’s algorithm, the most frequent measurements yield fractions $s/r$, where $r$ is the period (or order). Classical post-processing (e.g., continued fractions) extracts $k$. For this toy example, the simplified circuit doesn’t fully simulate ECDLP, so results are illustrative rather than functional.

## 5. Conclusion

This guide outlines a quantum approach to solving ECDLP in ECDH using Qiskit, focusing on a small-scale curve $y^2 = x^3 + 7$. The example simplifies elliptic curve arithmetic, which remains a significant challenge for real implementations. Key limitations include:
- **Qubit Requirements**: Cracking SECP256K1 requires hundreds of logical qubits, far beyond current hardware (e.g., 256-bit keys need ~2000-3000 qubits with error correction).
- **Noise**: Real quantum devices introduce errors, necessitating fault-tolerant systems.
- **Circuit Complexity**: Full ECDLP circuits demand custom gates, not yet standard in Qiskit.

Future work involves optimizing these circuits, integrating true elliptic curve operations, and testing on emerging quantum hardware.

## 6. Citations

- [1] Shor, P. W. (1994). "Algorithms for Quantum Computation: Discrete Logarithms and Factoring." *Proceedings of the 35th Annual Symposium on Foundations of Computer Science*. IEEE. DOI: 10.1109/SFCS.1994.365700.
- [2] Silverman, J. H. (2009). *The Arithmetic of Elliptic Curves*. Springer. ISBN: 978-0-387-09493-9.
- [3] Qiskit Team. (2023). "Qiskit: An Open-Source Framework for Quantum Computing." *Qiskit Documentation*. Available at: https://qiskit.org/documentation/.
- [4] Proos, J., & Zalka, C. (2003). "Shor’s Discrete Logarithm Quantum Algorithm for Elliptic Curves." *Quantum Information & Computation*, 3(4), 317-344.
  
