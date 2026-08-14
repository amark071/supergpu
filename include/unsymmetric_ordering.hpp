#ifndef SUPERNODAL_GPU_INCLUDE_UNSYMMETRIC_ORDERING_HPP
#define SUPERNODAL_GPU_INCLUDE_UNSYMMETRIC_ORDERING_HPP

#include <vector>

struct UnsymmetricOrdering {
    // perm[new_column] is the original column; iperm[old_column] is its
    // position after the COLAMD permutation.
    std::vector<int> perm;
    std::vector<int> iperm;

    // A value-biased maximum-cardinality matching supplies the initial row
    // permutation. Numerical row pivoting still occurs during factorization.
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

// Compute a structural column/row ordering without CHOLMOD.
UnsymmetricOrdering computeColamdOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices);

// Numeric overload: biases the full structural matching toward strong entries
// and computes row/column equilibration factors.
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
