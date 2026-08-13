#ifndef SUPERNODAL_GPU_INCLUDE_METIS_ORDERING_HPP
#define SUPERNODAL_GPU_INCLUDE_METIS_ORDERING_HPP

#include <vector>

struct MetisOrdering {
    // perm[new] = old, iperm[old] = new.
    std::vector<int> perm;
    std::vector<int> iperm;
};

// Compute a zero-based fill-reducing ordering with METIS_NodeND.
MetisOrdering computeMetisOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices);

#endif
