// pqc_expectation.cpp
#include <cudaq.h>
#include <iostream>
#include <vector>

// Define a parameterized quantum kernel
// Takes a single double as a parameter for an RX rotation
struct SimplePQC {
    void operator()(double theta) __qpu__ {
        cudaq::qvector q(1); // Allocate 1 qubit
        rx(theta, q[0]);     // Apply RX rotation parameterized by theta
        // No explicit measurement (mz) is needed when using cudaq::observe,
        // as it calculates the expectation value of an operator.
    }
};

// For a 2-qubit example with a vector of parameters
struct TwoQubitPQC {
    void operator()(std::vector<double> params) __qpu__ {
        cudaq::qvector q(2);
        rx(params[0], q[0]);
        ry(params[1], q[1]);
        cx(q[0], q[1]); // Entangle
        rx(params[2], q[0]);
        ry(params[3], q[1]);
    }
};


int main() {
    // --- Single Qubit PQC Example ---
    std::cout << "--- Single Qubit PQC Example ---" << std::endl;
    SimplePQC single_qubit_pqc;
    auto observable_z0 = cudaq::spin::z(0); // Measure Pauli Z on qubit 0

    double theta1 = 0.5; // Radians
    double exp_val1 = cudaq::observe(single_qubit_pqc, observable_z0, theta1);
    std::cout << "Expectation value of Z0 for PQC(theta=" << theta1 << "): " << exp_val1 
              << " (Expected: cos(" << theta1 << ") = " << std::cos(theta1) << ")" << std::endl;

    double theta2 = M_PI / 2.0; // Pi/2
    double exp_val2 = cudaq::observe(single_qubit_pqc, observable_z0, theta2);
    std::cout << "Expectation value of Z0 for PQC(theta=" << theta2 << "): " << exp_val2
              << " (Expected: cos(" << theta2 << ") = " << std::cos(theta2) << ")" << std::endl;

    // --- Two Qubit PQC Example ---
    std::cout << "\n--- Two Qubit PQC Example ---" << std::endl;
    TwoQubitPQC two_qubit_pqc;
    // Define an observable, e.g., Z0 * Z1 (parity) or just Z0
    auto observable_zz = cudaq::spin::z(0) * cudaq::spin::z(1);
    auto observable_z0_two_q = cudaq::spin::z(0);


    std::vector<double> params1 = {0.1, 0.2, 0.3, 0.4};
    double exp_val_zz1 = cudaq::observe(two_qubit_pqc, observable_zz, params1);
    std::cout << "Expectation value of Z0*Z1 for TwoQubitPQC with params1: " << exp_val_zz1 << std::endl;
    
    double exp_val_z0_1 = cudaq::observe(two_qubit_pqc, observable_z0_two_q, params1);
    std::cout << "Expectation value of Z0 for TwoQubitPQC with params1: " << exp_val_z0_1 << std::endl;


    std::vector<double> params2 = {M_PI, 0.0, M_PI/2.0, 0.0};
    double exp_val_zz2 = cudaq::observe(two_qubit_pqc, observable_zz, params2);
    std::cout << "Expectation value of Z0*Z1 for TwoQubitPQC with params2: " << exp_val_zz2 << std::endl;

    double exp_val_z0_2 = cudaq::observe(two_qubit_pqc, observable_z0_two_q, params2);
    std::cout << "Expectation value of Z0 for TwoQubitPQC with params2: " << exp_val_z0_2 << std::endl;


    std::cout << "\nCUDA-Q PQC and expectation value calculation complete." << std::endl;
    return 0;
}