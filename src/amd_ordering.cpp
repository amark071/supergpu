#include "amd_ordering.hpp"

#include <amd.h>

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
}

} // 匿名命名空间

AmdOrdering computeAmdOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices)
{
    validateCSC(n, col_ptr, row_indices);

    AmdOrdering ordering;
    ordering.perm.resize(static_cast<std::size_t>(n));
    ordering.iperm.resize(static_cast<std::size_t>(n));

    double control[AMD_CONTROL];
    double info[AMD_INFO];
    amd_defaults(control);

    const int result = amd_order(
        n,
        col_ptr.data(),
        row_indices.data(),
        ordering.perm.data(),
        control,
        info);

    if (result != AMD_OK && result != AMD_OK_BUT_JUMBLED) {
        throw std::runtime_error("amd_order failed with code " + std::to_string(result));
    }

    std::vector<bool> seen(static_cast<std::size_t>(n), false);
    for (int new_index = 0; new_index < n; ++new_index) {
        const int old_index = ordering.perm[static_cast<std::size_t>(new_index)];
        if (old_index < 0 || old_index >= n ||
            seen[static_cast<std::size_t>(old_index)]) {
            throw std::runtime_error("amd_order returned an invalid permutation");
        }
        seen[static_cast<std::size_t>(old_index)] = true;
        ordering.iperm[static_cast<std::size_t>(old_index)] = new_index;
    }

    return ordering;
}
