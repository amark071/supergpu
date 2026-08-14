#ifndef SUPERNODAL_GPU_INCLUDE_UNSYMMETRIC_SYMBOLIC_HPP
#define SUPERNODAL_GPU_INCLUDE_UNSYMMETRIC_SYMBOLIC_HPP

#include <cstddef>
#include <vector>

struct UnsymmetricSymbolicResult {
    std::vector<int> column_parent;
    std::vector<int> column_count;

    // Supernode s owns columns [supernode_ptr[s], supernode_ptr[s + 1]).
    std::vector<int> supernode_ptr;
    std::vector<int> supernode_parent;

    // The structural envelope of each front. Owned columns are first.
    std::vector<int> front_ptr;
    std::vector<int> front_indices;

    std::size_t envelope_off_diagonal_nonzeros = 0;
    std::size_t symbolic_factor_nonzeros = 0;
};

// Analyze the symmetric structural envelope pattern(A)+pattern(A)^T. This is
// an in-project symbolic analysis and does not call CHOLMOD.
UnsymmetricSymbolicResult analyzeUnsymmetricSupernodes(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices);

#endif
