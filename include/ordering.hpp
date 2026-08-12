#ifndef SUPERNODAL_GPU_INCLUDE_ORDERING_HPP
#define SUPERNODAL_GPU_INCLUDE_ORDERING_HPP

#include "fill_in.hpp"

#include <vector>

enum class OrderingMethod {
    Metis,
    Amd
};

/** @brief METIS与AMD比较后选出的最终排序结果 */
struct OrderingResult {
    std::vector<int> perm;
    std::vector<int> iperm;
    OrderingMethod method;
    FillInStatistics metis_fill;
    FillInStatistics amd_fill;
    FillInStatistics selected_fill;
};

/**
 * @brief 分别计算METIS和AMD排列，选择符号填充元更少的一种
 *
 * 当两种排列的填充元数量相同时选择AMD。
 */
OrderingResult computeBestOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices);

/** @brief 返回排序方法名称 */
const char* orderingMethodName(OrderingMethod method);

/** @brief 对0-based CSC矩阵执行P*A*P^T对称置换 */
void applySymmetricPermutationCSC(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values,
    const std::vector<int>& iperm,
    std::vector<int>& ordered_col_ptr,
    std::vector<int>& ordered_row_indices,
    std::vector<float>& ordered_values);

#endif
