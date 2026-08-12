#ifndef SUPERNODAL_GPU_INCLUDE_METIS_ORDERING_HPP
#define SUPERNODAL_GPU_INCLUDE_METIS_ORDERING_HPP

#include <vector>

struct MetisOrdering {
    // perm[新下标]=旧下标，iperm[旧下标]=新下标
    std::vector<int> perm;
    std::vector<int> iperm;
};

/** @brief 使用METIS_NodeND计算0-based CSC矩阵的填充减少排列 */
MetisOrdering computeMetisOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices);

#endif
