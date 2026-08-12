#ifndef SUPERNODAL_GPU_INCLUDE_FILL_IN_HPP
#define SUPERNODAL_GPU_INCLUDE_FILL_IN_HPP

#include <cstddef>
#include <vector>

/** @brief 对称符号消元的填充元统计结果 */
struct FillInStatistics {
    std::size_t original_edges;
    std::size_t fill_entries;
    std::size_t factor_nonzeros;
    double fill_ratio;
};

/**
 * @brief 统计给定排列产生的符号填充元，不进行数值分解
 * @param iperm 0-based逆排列，满足iperm[旧下标]=新下标
 */
FillInStatistics computeSymbolicFillIn(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<int>& iperm);

#endif
