#include "ordering.hpp"

#include "amd_ordering.hpp"
#include "metis_ordering.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void validateMatrixAndPermutation(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values,
    const std::vector<int>& iperm)
{
    if (n < 0 || col_ptr.size() != static_cast<std::size_t>(n + 1)) {
        throw std::invalid_argument("invalid CSC dimensions");
    }
    if (col_ptr.empty() || col_ptr.front() != 0 ||
        col_ptr.back() != static_cast<int>(row_indices.size())) {
        throw std::invalid_argument("CSC column pointers must be zero-based");
    }
    if (values.size() != row_indices.size() ||
        iperm.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("values or permutation size does not match the matrix");
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

OrderingResult computeBestOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices)
{
    const MetisOrdering metis = computeMetisOrdering(n, col_ptr, row_indices);
    const AmdOrdering amd = computeAmdOrdering(n, col_ptr, row_indices);

    OrderingResult result;
    result.metis_fill = computeSymbolicFillIn(n, col_ptr, row_indices, metis.iperm);
    result.amd_fill = computeSymbolicFillIn(n, col_ptr, row_indices, amd.iperm);

    if (result.amd_fill.fill_entries <= result.metis_fill.fill_entries) {
        result.perm = amd.perm;
        result.iperm = amd.iperm;
        result.method = OrderingMethod::Amd;
        result.selected_fill = result.amd_fill;
    } else {
        result.perm = metis.perm;
        result.iperm = metis.iperm;
        result.method = OrderingMethod::Metis;
        result.selected_fill = result.metis_fill;
    }
    return result;
}

const char* orderingMethodName(OrderingMethod method)
{
    return method == OrderingMethod::Amd ? "AMD" : "METIS";
}

void applySymmetricPermutationCSC(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values,
    const std::vector<int>& iperm,
    std::vector<int>& ordered_col_ptr,
    std::vector<int>& ordered_row_indices,
    std::vector<float>& ordered_values)
{
    validateMatrixAndPermutation(n, col_ptr, row_indices, values, iperm);

    typedef std::pair<int, float> Entry;
    std::vector<std::vector<Entry> > columns(static_cast<std::size_t>(n));
    for (int old_col = 0; old_col < n; ++old_col) {
        const int new_col = iperm[static_cast<std::size_t>(old_col)];
        const int begin = col_ptr[static_cast<std::size_t>(old_col)];
        const int end = col_ptr[static_cast<std::size_t>(old_col + 1)];
        for (int k = begin; k < end; ++k) {
            const int old_row = row_indices[static_cast<std::size_t>(k)];
            if (old_row < 0 || old_row >= n) {
                throw std::invalid_argument("CSC row index is out of range");
            }
            const int new_row = iperm[static_cast<std::size_t>(old_row)];
            columns[static_cast<std::size_t>(new_col)].push_back(
                Entry(new_row, values[static_cast<std::size_t>(k)]));
        }
    }

    ordered_col_ptr.assign(static_cast<std::size_t>(n + 1), 0);
    ordered_row_indices.clear();
    ordered_values.clear();
    ordered_row_indices.reserve(row_indices.size());
    ordered_values.reserve(values.size());

    for (int col = 0; col < n; ++col) {
        std::vector<Entry>& entries = columns[static_cast<std::size_t>(col)];
        std::sort(entries.begin(), entries.end(),
                  [](const Entry& lhs, const Entry& rhs) {
                      return lhs.first < rhs.first;
                  });
        for (std::size_t i = 0; i < entries.size(); ++i) {
            ordered_row_indices.push_back(entries[i].first);
            ordered_values.push_back(entries[i].second);
        }
        ordered_col_ptr[static_cast<std::size_t>(col + 1)] =
            static_cast<int>(ordered_row_indices.size());
    }
}
