#ifndef SUPERNODAL_GPU_INCLUDE_UNSYMMETRIC_VERIFICATION_HPP
#define SUPERNODAL_GPU_INCLUDE_UNSYMMETRIC_VERIFICATION_HPP

#include "solution_verification.hpp"

#include <vector>

std::vector<float> generalMatrixVectorProduct(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values,
    const std::vector<float>& x,
    float* matrix_infinity_norm);

// Computes b - A*x with double-precision accumulation while retaining the
// FP32 storage used by the factorization and correction solve.
std::vector<float> generalResidual(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values,
    const std::vector<float>& rhs,
    const std::vector<float>& solution);

ResidualVerification verifyGeneralSolution(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values,
    const std::vector<float>& rhs,
    const std::vector<float>& solution,
    const std::vector<float>& expected_solution);

#endif
