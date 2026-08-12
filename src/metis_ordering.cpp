#include "metis_ordering.hpp"

#include <metis.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void validateCSC(int n,
                 const std::vector<int>& col_ptr,
                 const std::vector<int>& row_indices)
{
    if (n < 0 || col_ptr.size() != static_cast<std::size_t>(n + 1)) {
        throw std::invalid_argument("invalid CSC dimensions");
    }
    if (col_ptr.empty() || col_ptr.front() != 0 ||
        col_ptr.back() != static_cast<int>(row_indices.size())) {
        throw std::invalid_argument("CSC column pointers must be zero-based");
    }
    for (int col = 0; col < n; ++col) {
        if (col_ptr[col] > col_ptr[col + 1]) {
            throw std::invalid_argument("CSC column pointers are not monotonic");
        }
    }
    for (std::size_t k = 0; k < row_indices.size(); ++k) {
        if (row_indices[k] < 0 || row_indices[k] >= n) {
            throw std::invalid_argument("CSC row index is out of range");
        }
    }
}

} // 匿名命名空间

MetisOrdering computeMetisOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices)
{
    validateCSC(n, col_ptr, row_indices);

    std::vector<std::vector<idx_t> > graph(static_cast<std::size_t>(n));
    for (int col = 0; col < n; ++col) {
        const int begin = col_ptr[col];
        const int end = col_ptr[col + 1];
        for (int k = begin; k < end; ++k) {
            const int row = row_indices[static_cast<std::size_t>(k)];
            if (row != col) {
                graph[static_cast<std::size_t>(col)].push_back(static_cast<idx_t>(row));
                graph[static_cast<std::size_t>(row)].push_back(static_cast<idx_t>(col));
            }
        }
    }

    std::vector<idx_t> xadj(static_cast<std::size_t>(n + 1), 0);
    for (int vertex = 0; vertex < n; ++vertex) {
        std::vector<idx_t>& neighbors = graph[static_cast<std::size_t>(vertex)];
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        xadj[static_cast<std::size_t>(vertex + 1)] =
            xadj[static_cast<std::size_t>(vertex)] + static_cast<idx_t>(neighbors.size());
    }

    std::vector<idx_t> adjncy(static_cast<std::size_t>(xadj.back()));
    for (int vertex = 0; vertex < n; ++vertex) {
        std::copy(graph[static_cast<std::size_t>(vertex)].begin(),
                  graph[static_cast<std::size_t>(vertex)].end(),
                  adjncy.begin() + xadj[static_cast<std::size_t>(vertex)]);
    }

    idx_t vertex_count = static_cast<idx_t>(n);
    std::vector<idx_t> metis_perm(static_cast<std::size_t>(n));
    std::vector<idx_t> metis_iperm(static_cast<std::size_t>(n));
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;

    const int result = METIS_NodeND(
        &vertex_count, xadj.data(), adjncy.data(), nullptr,
        options, metis_perm.data(), metis_iperm.data());
    if (result != METIS_OK) {
        throw std::runtime_error("METIS_NodeND failed with code " + std::to_string(result));
    }

    MetisOrdering ordering;
    ordering.perm.resize(static_cast<std::size_t>(n));
    ordering.iperm.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        ordering.perm[static_cast<std::size_t>(i)] =
            static_cast<int>(metis_perm[static_cast<std::size_t>(i)]);
        ordering.iperm[static_cast<std::size_t>(i)] =
            static_cast<int>(metis_iperm[static_cast<std::size_t>(i)]);
    }
    return ordering;
}
