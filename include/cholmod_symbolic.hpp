#ifndef SUPERNODAL_GPU_INCLUDE_CHOLMOD_SYMBOLIC_HPP
#define SUPERNODAL_GPU_INCLUDE_CHOLMOD_SYMBOLIC_HPP

#include <string>
#include <vector>

/** @brief CHOLMOD 生成的对称矩阵基本超节点符号分析结果。 */
struct CholmodSymbolicResult {
    // 列消去树：column_parent[j] 是列 j 的父列，-1 表示根。
    std::vector<int> column_parent;

    // 符号因子 L 每列的非零元数，包括对角元。
    std::vector<int> column_count;

    // 超节点 s 包含列 [supernode_ptr[s], supernode_ptr[s + 1])。
    std::vector<int> supernode_ptr;

    // 超节点树：supernode_parent[s] 是其父超节点，-1 表示根。
    std::vector<int> supernode_parent;

    // 超节点 s 的行结构位于
    // supernode_rows[row_ptr[s] ... row_ptr[s + 1])。
    std::vector<int> row_ptr;
    std::vector<int> supernode_rows;
};

/**
 * @brief 对已重排的 0-based CSC 对称结构构造消去树和基本超节点。
 *
 * 输入可以存储下三角、上三角或完整对称结构；函数会只保留一份
 * 去重的上三角结构交给 CHOLMOD。数值不参与分析。
 */
CholmodSymbolicResult analyzeBasicSupernodesWithCholmod(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices);

/** @brief 导出符号因子和消去树可视化所需的 CSV 数据。 */
void writeSymbolicVisualizationData(
    const std::string& output_directory,
    int n,
    const CholmodSymbolicResult& symbolic);

#endif
