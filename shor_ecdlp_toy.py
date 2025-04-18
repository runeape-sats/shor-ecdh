# shor_ecdlp_toy.py
#!/usr/bin/env python3

"""
Theoretical Implementation of Shor's Algorithm for ECDLP over F_17 using Qiskit.

Curve: y^2 = x^3 + 7 (mod 17), G = (5,8), Q = 2G = (5,9), subgroup order 3.
Aims to find k where Q = kG (k = 2).

NOTE: The quantum oracle is a conceptual placeholder due to classical simulation limits.
A true solution requires a quantum computer with EC arithmetic circuits. On a real quantum computer with 
a proper oracle, it would recover 𝑘=2 efficiently. For now, it’s a pedagogical tool showing Shor’s algorithm’s 
framework. To go further, you’d need quantum hardware and a detailed EC arithmetic implementation—beyond 
current classical simulation capabilities.
"""

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.qasm3 import dumps
    import numpy as np
    print("Qiskit, Qiskit Aer, and NumPy imported successfully.")
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please install required libraries: pip install qiskit qiskit-aer numpy")
    exit(1)

# Elliptic Curve Parameters
p = 17      # Prime modulus for F_17
a = 0       # Curve parameter 'a'
b = 7       # Curve parameter 'b'
G = (5, 8)  # Base point
Q = (5, 9)  # Public key Q = 2G, k = 2
order = 3   # Subgroup order
print(f"\nElliptic Curve Defined: y^2 = x^3 + {a}x + {b} (mod {p})")
print(f"Base Point G = {G}")
print(f"Public Key Q = {Q}")
print(f"Subgroup Order = {order}")

# Verify points
def is_point_on_curve(point, p_mod, a_param, b_param):
    if point is None:
        return True
    x, y = point
    lhs = (y**2) % p_mod
    rhs = (x**3 + a_param * x + b_param) % p_mod
    return lhs == rhs

for point, label in [(G, "G"), (Q, "Q")]:
    is_on_curve = is_point_on_curve(point, p, a, b)
    print(f"Is point {label} {point} on the curve? {'Yes' if is_on_curve else 'No'}")
    if not is_on_curve:
        print(f"Error: Point {label} is not on the curve. Exiting.")
        exit(1)

# Classical EC Point Multiplication (for verification)
def ec_point_mult(k, point, p_mod, a_param):
    if k == 0:
        return None
    result = point
    for _ in range(k - 1):
        if result is None:
            return None
        x1, y1 = result
        lambda_val = (3 * x1**2) % p_mod * pow(2 * y1, -1, p_mod) % p_mod
        x2 = (lambda_val**2 - 2 * x1) % p_mod
        y2 = (lambda_val * (x1 - x2) - y1) % p_mod
        result = (x2, y2)
    return result

# Quantum Fourier Transform
def qft(n):
    qc = QuantumCircuit(n, name="QFT")
    for i in range(n - 1, -1, -1):
        qc.h(i)
        for j in range(i - 1, -1, -1):
            qc.cp(np.pi / (2**(i - j)), j, i)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc

# Shor's Algorithm for ECDLP
def shors_dlog_true(n_control_qubits):
    # Define qubit allocation
    n_target_qubits = 2  # Simplified: encode 3 states (O, (5,8), (5,9)) with 2 qubits
    total_qubits = 2 * n_control_qubits + n_target_qubits  # a, b registers + target
    qc = QuantumCircuit(total_qubits, 2 * n_control_qubits, name="ShorECDLP_True")

    # Step 1: Superposition on control registers (a: 0 to n-1, b: n to 2n-1)
    qc.h(range(2 * n_control_qubits))

    # Step 2: Quantum Oracle (Placeholder for f(a, b) = (a + 2b)G mod 3)
    print("\nImplementing quantum oracle (conceptual placeholder)...")
    # Subgroup: O -> |00>, (5,8) -> |01>, (5,9) -> |10>
    k_secret = 2  # Hardcoded for Q = 2G; in reality, this is what we solve for
    for a in range(2**n_control_qubits):
        for b in range(2**n_control_qubits):
            val = (a + k_secret * b) % order
            control_qubits = [i for i in range(n_control_qubits) if (a >> i) & 1] + \
                            [i + n_control_qubits for i in range(n_control_qubits) if (b >> i) & 1]
            target_base = 2 * n_control_qubits
            if val == 1:  # (5,8)
                if control_qubits:
                    for ctrl in control_qubits:
                        qc.cx(ctrl, target_base + 1)  # Set to |01>
            elif val == 2:  # (5,9)
                if control_qubits:
                    for ctrl in control_qubits:
                        qc.cx(ctrl, target_base)  # Set to |10>

    qc.barrier()

    # Step 3: Inverse QFT on control registers
    print(f"Applying Inverse QFT to control registers...")
    qft_gate = qft(n_control_qubits)
    qc.compose(qft_gate.inverse(), range(n_control_qubits), inplace=True)  # a register
    qc.compose(qft_gate.inverse(), range(n_control_qubits, 2 * n_control_qubits), inplace=True)  # b register

    # Step 4: Measure control registers
    qc.measure(range(2 * n_control_qubits), range(2 * n_control_qubits))

    return qc

# Classical Post-Processing
def post_process_measurements(counts, order=3, n_qubits=2):
    most_common = max(counts, key=counts.get)
    a_str, b_str = most_common[:n_qubits], most_common[n_qubits:]
    a_measured = int(a_str, 2)
    b_measured = int(b_str, 2)
    print(f"\nMost common measurement: a' = {a_measured} (binary: {a_str}), b' = {b_measured} (binary: {b_str})")
    
    # In a real quantum run, use continued fractions to find r and solve for k
    # Simplified here: test possible k values
    for k in range(order):
        if (a_measured + k * b_measured) % order == 0 and b_measured != 0:
            print(f"Possible k = {k} satisfies (a' + k * b') mod {order} = 0")
            return k
    print("Could not determine k directly (b' = 0 or no solution).")
    return None

# Main Execution
if __name__ == "__main__":
    print("\n--- Starting True ECDLP Solver Simulation (mod 17) ---")
    num_control_qubits = 2  # 4 states cover order 3
    print(f"Using {num_control_qubits} qubits per control register.")

    print("Building the quantum circuit...")
    circuit = shors_dlog_true(num_control_qubits)
    print("\nCircuit Diagram:")
    print(circuit.draw(output='text', fold=-1))

    backend = AerSimulator()
    num_shots = 1024
    print(f"Running simulation with {num_shots} shots...")
    
    print("Transpiling circuit for AerSimulator...")
    transpiled_circuit = transpile(circuit, backend)
    
    job = backend.run(transpiled_circuit, shots=num_shots)
    result = job.result()
    counts = result.get_counts()
    print("\nMeasurement counts:")
    sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    print(sorted_counts)

    print("\n--- Post-Processing ---")
    k = post_process_measurements(sorted_counts, order=order, n_qubits=num_control_qubits)
    if k is not None:
        print(f"Estimated discrete logarithm k = {k}")
        print(f"Verification: {k}G = {ec_point_mult(k, G, p, a)} should equal Q = {Q}")
    else:
        print("Post-processing failed to determine k.")

    print("\n--- Notes ---")
    print("This is a theoretical demo. The oracle is a placeholder hardcoded with k=2.")
    print("A true ECDLP solution requires:")
    print("1. A quantum oracle implementing EC point multiplication.")
    print("2. A quantum computer with sufficient qubits and low noise.")
    print("3. Classical continued fractions for large orders.")
    print("\n--- Script Finished ---")

    print("\n\n--- Export the circuit to an OpenQASM 3.0 string ---")
    qasm3_output = dumps(circuit)
    print(qasm3_output)
