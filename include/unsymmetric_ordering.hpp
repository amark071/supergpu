#ifndef SUPERNODAL_GPU_INCLUDE_UNSYMMETRIC_ORDERING_HPP
#define SUPERNODAL_GPU_INCLUDE_UNSYMMETRIC_ORDERING_HPP

#include <vector>

struct UnsymmetricOrdering {
    // perm[new_column] is the original column; iperm[old_column] is its
    // position after the COLAMD permutation.
    std::vector<int> perm;
    std::vector<int> iperm;

    // A structural maximum matching supplies an initial row permutation.
    // Numerical row pivoting is still performed during factorization.
    std::vector<int> row_perm;
    std::vector<int> row_iperm;

    int structural_rank = 0;
};

struct UnsymmetricPermutedCsc {
    std::vector<int> col_ptr;
    std::vector<int> row_indices;
    std::vector<float> values;
};

// Compute a column ordering for a general sparse matrix without CHOLMOD.
UnsymmetricOrdering computeColamdOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices);

// Form A(row_perm, perm) in the common reordered row/column coordinate system.
UnsymmetricPermutedCsc applyUnsymmetricPermutationCsc(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values,
    const UnsymmetricOrdering& ordering);

#endif
