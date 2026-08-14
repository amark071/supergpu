#include "unsymmetric_ordering.hpp"

#include "amd_ordering.hpp"
#include "unsymmetric_matching.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

void validateCsc(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>* values)
{
    if (n < 0 || col_ptr.size() != static_cast<std::size_t>(n + 1) ||
        col_ptr.empty() || col_ptr.front() != 0 ||
        col_ptr.back() != static_cast<int>(row_indices.size())) {
        throw std::invalid_argument("invalid zero-based CSC matrix");
    }
    if (values != 0 && values->size() != row_indices.size()) {
        throw std::invalid_argument("CSC values and row indices differ in size");
    }
    for (int col = 0; col < n; ++col) {
        if (col_ptr[static_cast<std::size_t>(col)] >
            col_ptr[static_cast<std::size_t>(col + 1)]) {
            throw std::invalid_argument("CSC column pointers are not monotonic");
        }
    }
    for (std::size_t p = 0; p < row_indices.size(); ++p) {
        if (row_indices[p] < 0 || row_indices[p] >= n) {
            throw std::invalid_argument("CSC row index is out of range");
        }
    }
}

void validatePermutation(int n, const UnsymmetricOrdering& ordering)
{
    if (ordering.perm.size() != static_cast<std::size_t>(n) ||
        ordering.iperm.size() != static_cast<std::size_t>(n) ||
        ordering.row_perm.size() != static_cast<std::size_t>(n) ||
        ordering.row_iperm.size() != static_cast<std::size_t>(n) ||
        ordering.row_scale.size() != static_cast<std::size_t>(n) ||
        ordering.col_scale.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("unsymmetric permutation size mismatch");
    }
    std::vector<unsigned char> seen(static_cast<std::size_t>(n), 0);
    for (int new_col = 0; new_col < n; ++new_col) {
        const int old_col = ordering.perm[static_cast<std::size_t>(new_col)];
        if (old_col < 0 || old_col >= n || seen[static_cast<std::size_t>(old_col)] ||
            ordering.iperm[static_cast<std::size_t>(old_col)] != new_col) {
            throw std::invalid_argument("invalid unsymmetric column permutation");
        }
        seen[static_cast<std::size_t>(old_col)] = 1;
    }
    std::fill(seen.begin(), seen.end(), 0);
    for (int new_row = 0; new_row < n; ++new_row) {
        const int old_row = ordering.row_perm[static_cast<std::size_t>(new_row)];
        if (old_row < 0 || old_row >= n || seen[static_cast<std::size_t>(old_row)] ||
            ordering.row_iperm[static_cast<std::size_t>(old_row)] != new_row) {
            throw std::invalid_argument("invalid unsymmetric row permutation");
        }
        seen[static_cast<std::size_t>(old_row)] = 1;
    }
}

std::vector<int> completeMatchedRows(
    int n,
    const std::vector<int>& column_to_row)
{
    std::vector<int> row_per_coordinate(static_cast<std::size_t>(n), -1);
    std::vector<unsigned char> used(static_cast<std::size_t>(n), 0);
    for (int col = 0; col < n; ++col) {
        const int row = column_to_row[static_cast<std::size_t>(col)];
        if (row >= 0 && row < n && !used[static_cast<std::size_t>(row)]) {
            row_per_coordinate[static_cast<std::size_t>(col)] = row;
            used[static_cast<std::size_t>(row)] = 1;
        }
    }
    std::vector<int> unused_rows;
    unused_rows.reserve(static_cast<std::size_t>(n));
    for (int row = 0; row < n; ++row) {
        if (!used[static_cast<std::size_t>(row)]) {
            unused_rows.push_back(row);
        }
    }
    std::size_t unused = 0;
    for (int coordinate = 0; coordinate < n; ++coordinate) {
        if (row_per_coordinate[static_cast<std::size_t>(coordinate)] < 0) {
            row_per_coordinate[static_cast<std::size_t>(coordinate)] =
                unused_rows[unused++];
        }
    }
    return row_per_coordinate;
}

std::vector<int> invertPermutation(const std::vector<int>& permutation)
{
    std::vector<int> inverse(permutation.size(), -1);
    for (std::size_t new_index = 0; new_index < permutation.size(); ++new_index) {
        inverse[static_cast<std::size_t>(permutation[new_index])] =
            static_cast<int>(new_index);
    }
    return inverse;
}

AmdOrdering orderMatchedPattern(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<int>& matched_row_inverse)
{
    std::vector<int> matched_rows(row_indices.size());
    for (std::size_t p = 0; p < row_indices.size(); ++p) {
        matched_rows[p] = matched_row_inverse[
            static_cast<std::size_t>(row_indices[p])];
    }
    return computeAmdOrdering(n, col_ptr, matched_rows);
}

UnsymmetricOrdering composeOrdering(
    int n,
    const UnsymmetricMatching& matching,
    const std::vector<int>& matched_rows,
    const AmdOrdering& amd)
{
    UnsymmetricOrdering result;
    result.structural_rank = matching.cardinality;
    result.perm.resize(static_cast<std::size_t>(n));
    result.iperm.resize(static_cast<std::size_t>(n));
    result.row_perm.resize(static_cast<std::size_t>(n));
    result.row_iperm.resize(static_cast<std::size_t>(n));
    result.row_scale.resize(static_cast<std::size_t>(n), 1.0f);
    result.col_scale.resize(static_cast<std::size_t>(n), 1.0f);
    for (int new_coordinate = 0; new_coordinate < n; ++new_coordinate) {
        const int matched_coordinate =
            amd.perm[static_cast<std::size_t>(new_coordinate)];
        const int old_col = matched_coordinate;
        const int old_row =
            matched_rows[static_cast<std::size_t>(matched_coordinate)];
        result.perm[static_cast<std::size_t>(new_coordinate)] = old_col;
        result.iperm[static_cast<std::size_t>(old_col)] = new_coordinate;
        result.row_perm[static_cast<std::size_t>(new_coordinate)] = old_row;
        result.row_iperm[static_cast<std::size_t>(old_row)] = new_coordinate;
        result.row_scale[static_cast<std::size_t>(new_coordinate)] =
            matching.row_scale[static_cast<std::size_t>(old_row)];
        result.col_scale[static_cast<std::size_t>(new_coordinate)] =
            matching.column_scale[static_cast<std::size_t>(old_col)];
    }
    return result;
}

} // namespace

UnsymmetricOrdering computeMatchingAmdOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values)
{
    validateCsc(n, col_ptr, row_indices, &values);
    const UnsymmetricMatching matching = computeDuffKosterMatching(
        n, col_ptr, row_indices, values);
    const std::vector<int> matched_rows = completeMatchedRows(
        n, matching.column_to_row);
    const std::vector<int> matched_row_inverse =
        invertPermutation(matched_rows);

    AmdOrdering amd;
    if (n > 0) {
        amd = orderMatchedPattern(
            n, col_ptr, row_indices, matched_row_inverse);
    }
    const UnsymmetricOrdering result = composeOrdering(
        n, matching, matched_rows, amd);
    validatePermutation(n, result);
    return result;
}

UnsymmetricOrdering computeAmdOnlyOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices)
{
    validateCsc(n, col_ptr, row_indices, 0);
    AmdOrdering amd;
    if (n > 0) {
        amd = computeAmdOrdering(n, col_ptr, row_indices);
    }
    UnsymmetricOrdering result;
    result.perm = amd.perm;
    result.iperm = amd.iperm;
    result.row_perm = amd.perm;
    result.row_iperm = amd.iperm;
    result.row_scale.assign(static_cast<std::size_t>(n), 1.0f);
    result.col_scale.assign(static_cast<std::size_t>(n), 1.0f);
    result.structural_rank = -1;
    validatePermutation(n, result);
    return result;
}

UnsymmetricOrdering computeMatchingAmdOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices)
{
    return computeMatchingAmdOrdering(
        n, col_ptr, row_indices,
        std::vector<float>(row_indices.size(), 1.0f));
}

UnsymmetricOrdering computeColamdOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values)
{
    return computeMatchingAmdOrdering(n, col_ptr, row_indices, values);
}

UnsymmetricOrdering computeColamdOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices)
{
    return computeMatchingAmdOrdering(n, col_ptr, row_indices);
}

UnsymmetricPermutedCsc applyUnsymmetricPermutationCsc(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values,
    const UnsymmetricOrdering& ordering)
{
    validateCsc(n, col_ptr, row_indices, &values);
    validatePermutation(n, ordering);

    UnsymmetricPermutedCsc result;
    result.col_ptr.resize(static_cast<std::size_t>(n + 1), 0);
    result.row_indices.reserve(row_indices.size());
    result.values.reserve(values.size());
    for (int new_col = 0; new_col < n; ++new_col) {
        const int old_col = ordering.perm[static_cast<std::size_t>(new_col)];
        const int begin = col_ptr[static_cast<std::size_t>(old_col)];
        const int end = col_ptr[static_cast<std::size_t>(old_col + 1)];
        for (int p = begin; p < end; ++p) {
            const int old_row = row_indices[static_cast<std::size_t>(p)];
            const int new_row =
                ordering.row_iperm[static_cast<std::size_t>(old_row)];
            result.row_indices.push_back(new_row);
            const double scaled =
                static_cast<double>(values[static_cast<std::size_t>(p)]) *
                ordering.row_scale[static_cast<std::size_t>(new_row)] *
                ordering.col_scale[static_cast<std::size_t>(new_col)];
            result.values.push_back(static_cast<float>(scaled));
        }
        result.col_ptr[static_cast<std::size_t>(new_col + 1)] =
            static_cast<int>(result.row_indices.size());
    }
    return result;
}
