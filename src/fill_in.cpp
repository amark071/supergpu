#include "fill_in.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace {

void validateInput(int n,
                   const std::vector<int>& col_ptr,
                   const std::vector<int>& row_indices,
                   const std::vector<int>& iperm)
{
    if (n < 0 || col_ptr.size() != static_cast<std::size_t>(n + 1)) {
        throw std::invalid_argument("invalid CSC dimensions");
    }
    if (col_ptr.empty() || col_ptr.front() != 0 ||
        col_ptr.back() != static_cast<int>(row_indices.size())) {
        throw std::invalid_argument("CSC column pointers must be zero-based");
    }
    for (int col = 0; col < n; ++col) {
        if (col_ptr[static_cast<std::size_t>(col)] >
            col_ptr[static_cast<std::size_t>(col + 1)]) {
            throw std::invalid_argument("CSC column pointers are not monotonic");
        }
    }
    for (std::size_t i = 0; i < row_indices.size(); ++i) {
        if (row_indices[i] < 0 || row_indices[i] >= n) {
            throw std::invalid_argument("CSC row index is out of range");
        }
    }
    if (iperm.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("permutation size does not match the matrix");
    }

    std::vector<bool> seen(static_cast<std::size_t>(n), false);
    for (int old_index = 0; old_index < n; ++old_index) {
        const int new_index = iperm[static_cast<std::size_t>(old_index)];
        if (new_index < 0 || new_index >= n ||
            seen[static_cast<std::size_t>(new_index)]) {
            throw std::invalid_argument("iperm is not a valid permutation");
        }
        seen[static_cast<std::size_t>(new_index)] = true;
    }
}

} // 匿名命名空间

FillInStatistics computeSymbolicFillIn(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<int>& iperm)
{
    validateInput(n, col_ptr, row_indices, iperm);

    // 按照新编号构造A+A^T的严格上三角结构，每条无向边只存储一次。
    std::vector<std::vector<int> > upper_columns(static_cast<std::size_t>(n));
    for (int old_col = 0; old_col < n; ++old_col) {
        const int new_col = iperm[static_cast<std::size_t>(old_col)];
        const int begin = col_ptr[static_cast<std::size_t>(old_col)];
        const int end = col_ptr[static_cast<std::size_t>(old_col + 1)];
        for (int k = begin; k < end; ++k) {
            const int old_row = row_indices[static_cast<std::size_t>(k)];
            const int new_row = iperm[static_cast<std::size_t>(old_row)];
            if (new_row != new_col) {
                const int upper_col = new_row > new_col ? new_row : new_col;
                const int upper_row = new_row > new_col ? new_col : new_row;
                upper_columns[static_cast<std::size_t>(upper_col)].push_back(upper_row);
            }
        }
    }

    std::size_t original_edges = 0;
    for (int col = 0; col < n; ++col) {
        std::vector<int>& rows = upper_columns[static_cast<std::size_t>(col)];
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        original_edges += rows.size();
    }

    // 根据置换后的上三角结构建立消元树。
    std::vector<int> parent(static_cast<std::size_t>(n), -1);
    std::vector<int> ancestor(static_cast<std::size_t>(n), -1);
    for (int col = 0; col < n; ++col) {
        const std::vector<int>& rows = upper_columns[static_cast<std::size_t>(col)];
        for (std::size_t p = 0; p < rows.size(); ++p) {
            int row = rows[p];
            while (row != -1 && row < col) {
                const int next = ancestor[static_cast<std::size_t>(row)];
                ancestor[static_cast<std::size_t>(row)] = col;
                if (next == -1) {
                    parent[static_cast<std::size_t>(row)] = col;
                }
                row = next;
            }
        }
    }

    // 对每一行求消元树可达集，其大小就是L该行严格下三角的非零元数。
    std::size_t factor_off_diagonal = 0;
    std::vector<int> marked(static_cast<std::size_t>(n), -1);
    for (int col = 0; col < n; ++col) {
        const std::vector<int>& rows = upper_columns[static_cast<std::size_t>(col)];
        for (std::size_t p = 0; p < rows.size(); ++p) {
            int row = rows[p];
            while (row != -1 && row < col &&
                   marked[static_cast<std::size_t>(row)] != col) {
                marked[static_cast<std::size_t>(row)] = col;
                ++factor_off_diagonal;
                row = parent[static_cast<std::size_t>(row)];
            }
        }
    }

    const std::size_t fill_entries = factor_off_diagonal - original_edges;

    FillInStatistics statistics;
    statistics.original_edges = original_edges;
    statistics.fill_entries = fill_entries;
    statistics.factor_nonzeros =
        static_cast<std::size_t>(n) + factor_off_diagonal;

    const std::size_t original_triangular_nonzeros =
        static_cast<std::size_t>(n) + original_edges;
    statistics.fill_ratio = original_triangular_nonzeros == 0
        ? 1.0
        : static_cast<double>(statistics.factor_nonzeros) /
              static_cast<double>(original_triangular_nonzeros);
    return statistics;
}
