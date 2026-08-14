#ifndef SUPERNODAL_GPU_INCLUDE_UNSYMMETRIC_MATCHING_HPP
#define SUPERNODAL_GPU_INCLUDE_UNSYMMETRIC_MATCHING_HPP

#include <vector>

struct UnsymmetricMatching {
    int cardinality = 0;
    bool perfect = false;

    // row_to_column[old_row] and column_to_row[old_column] describe the
    // weighted bipartite matching in the original matrix coordinates.
    std::vector<int> row_to_column;
    std::vector<int> column_to_row;

    // Duff--Koster diagonal scaling for Dr * A * Dc.
    std::vector<float> row_scale;
    std::vector<float> column_scale;
};

// Compute a maximum-cardinality matching that maximizes the product of the
// matched entry magnitudes. This is the unsymmetric Duff--Koster/MC64-style
// preprocessing used before the fill-reducing ordering.
UnsymmetricMatching computeDuffKosterMatching(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values);

#endif
