#ifndef SUPERNODAL_GPU_INCLUDE_UNSYMMETRIC_ORDERING_HPP
#define SUPERNODAL_GPU_INCLUDE_UNSYMMETRIC_ORDERING_HPP

#include <vector>

struct UnsymmetricOrdering {
    // perm[new_column] is the original column; iperm[old_column] is its
    // position after matching followed by AMD.
    std::vector<int> perm;
    std::vector<int> iperm;

    // Duff--Koster weighted matching supplies the initial row permutation.
    // Numerical row pivoting still occurs during factorization.
    std::vector<int> row_perm;
    std::vector<int> row_iperm;

    // Diagonal equilibration for Dr * P * A * Q * Dc.
    std::vector<float> row_scale;
    std::vector<float> col_scale;

    int structural_rank = 0;
};

struct UnsymmetricPermutedCsc {
    std::vector<int> col_ptr;
    std::vector<int> row_indices;
    std::vector<float> values;
};

// Compute Duff--Koster weighted matching and scaling first, then apply AMD to
// the matched matrix's symmetric structural envelope. CHOLMOD is not used.
UnsymmetricOrdering computeMatchingAmdOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices);

UnsymmetricOrdering computeMatchingAmdOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values);

// Source-compatible aliases retained for callers of the earlier interface.
// These now execute matching followed by AMD; COLAMD is no longer used.
UnsymmetricOrdering computeColamdOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices);

UnsymmetricOrdering computeColamdOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values);

// Form A(row_perm, perm) in the common reordered row/column coordinate system.
UnsymmetricPermutedCsc applyUnsymmetricPermutationCsc(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values,
    const UnsymmetricOrdering& ordering);

#endif
