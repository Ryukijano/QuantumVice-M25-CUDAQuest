// bell_state_cudaq.cpp
#include <cudaq.h>
#include <iostream>

// Define the quantum kernel for Bell state preparation and measurement
struct BellKernel {
    // The __qpu__ attribute indicates this is a quantum kernel
    void operator()() __qpu__ {
        // Allocate 2 qubits using a qvector
        cudaq::qvector q(2);

        // Apply Hadamard gate to the first qubit (q[0])
        // This puts q[0] into a superposition state (|0> + |1>) / sqrt(2)
        h(q[0]);

        // Apply CNOT (Controlled-X) gate
        // q[0] is the control qubit, q[1] is the target qubit
        // If q[0] is |1>, q[1] is flipped. If q[0] is |0>, q[1] is unchanged.
        // This entangles the two qubits, creating the Bell state (|00> + |11>) / sqrt(2)
        cx(q[0], q[1]);

        // Measure all qubits in the register 'q'
        // This collapses the superposition into a classical bitstring ("00" or "11")
        mz(q);
    }
};

int main() {
    std::cout << "Attempting to sample BellKernel using CUDA-Q..." << std::endl;
    
    // Instantiate the kernel
    BellKernel bell_kernel;

    // Sample the results (default 1000 shots).
    // This will execute the quantum kernel, simulating the measurements.
    // If compiled with --target nvidia, cuQuantum will be used for acceleration.
    auto counts = cudaq::sample(bell_kernel);

    std::cout << "\nBell State Measurement Counts (Expected: ~50% '00', ~50% '11'):" << std::endl;
    // Iterate through the measurement results and print them.
    // `counts` is a cudaq::sample_result, which behaves like a map from bitstring to count.
    for (auto& [bitstring, count] : counts) {
        std::cout << "  Observed state: \"" << bitstring << "\"  - Count: " << count << std::endl;
    }
    // For a more direct print of the counts object:
    // counts.dump();

    std::cout << "\nCUDA-Q Bell state kernel execution complete." << std::endl;
    return 0;
}