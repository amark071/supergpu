#ifndef SUPERNODAL_GPU_INCLUDE_UNSYMMETRIC_VERIFICATION_HPP
#define SUPERNODAL_GPU_INCLUDE_UNSYMMETRIC_VERIFICATION_HPP

#include "solution_verification.hpp"

#include <vector>

std::vector<double> generalMatrixVectorProduct(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<double>& values,
    const std::vector<double>& x,
    double* matrix_infinity_norm);

// Computes b - A*x entirely in FP64.
std::vector<double> generalResidual(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<double>& values,
    const std::vector<double>& rhs,
    const std::vector<double>& solution);

ResidualVerification verifyGeneralSolution(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<double>& values,
    const std::vector<double>& rhs,
    const std::vector<double>& solution,
    const std::vector<double>& expected_solution);

#endif
