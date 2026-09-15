#ifndef SUPERNODAL_GPU_INCLUDE_SOLUTION_VERIFICATION_HPP
#define SUPERNODAL_GPU_INCLUDE_SOLUTION_VERIFICATION_HPP

#include <cstdint>
#include <unordered_map>
#include <vector>

struct SymmetricEntryAccumulator {
    SymmetricEntryAccumulator();

    float lower;
    float upper;
    int lower_count;
    int upper_count;
};

using SymmetricEntries =
    std::unordered_map<std::uint64_t, SymmetricEntryAccumulator>;

struct ResidualVerification {
    double matrix_infinity_norm;
    double rhs_infinity_norm;
    double solution_infinity_norm;
    double absolute_residual;
    double relative_residual;
    double scaled_backward_error;
    double relative_solution_error;
};

SymmetricEntries buildSymmetricEntries(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values);

std::vector<float> symmetricMatrixVectorProduct(
    int n,
    const SymmetricEntries& entries,
    bool symmetrize_input,
    const std::vector<float>& x,
    float* matrix_infinity_norm);

ResidualVerification verifySolution(
    int n,
    const SymmetricEntries& entries,
    bool symmetrize_input,
    const std::vector<float>& rhs,
    const std::vector<float>& solution,
    const std::vector<float>& expected_solution);

#endif
