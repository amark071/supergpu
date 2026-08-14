#include "unsymmetric_ordering.hpp"

#include <colamd.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using MatchingAdjacency = std::vector<std::vector<int> >;

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
        throw std::invalid_argument("COLAMD permutation size mismatch");
    }
    std::vector<unsigned char> seen(static_cast<std::size_t>(n), 0);
    for (int new_col = 0; new_col < n; ++new_col) {
        const int old_col = ordering.perm[static_cast<std::size_t>(new_col)];
        if (old_col < 0 || old_col >= n || seen[static_cast<std::size_t>(old_col)]) {
            throw std::invalid_argument("invalid COLAMD permutation");
        }
        seen[static_cast<std::size_t>(old_col)] = 1;
        if (ordering.iperm[static_cast<std::size_t>(old_col)] != new_col) {
            throw std::invalid_argument("COLAMD inverse permutation mismatch");
        }
    }
    std::fill(seen.begin(), seen.end(), 0);
    for (int new_row = 0; new_row < n; ++new_row) {
        const int old_row = ordering.row_perm[static_cast<std::size_t>(new_row)];
        if (old_row < 0 || old_row >= n || seen[static_cast<std::size_t>(old_row)] ||
            ordering.row_iperm[static_cast<std::size_t>(old_row)] != new_row) {
            throw std::invalid_argument("invalid structural row permutation");
        }
        seen[static_cast<std::size_t>(old_row)] = 1;
    }
}

bool augmentColumnLayered(
    int new_col,
    const MatchingAdjacency& adjacency,
    std::vector<int>& col_match,
    std::vector<int>& row_match,
    std::vector<int>& distance)
{
    const std::vector<int>& rows = adjacency[static_cast<std::size_t>(new_col)];
    for (std::size_t p = 0; p < rows.size(); ++p) {
        const int row = rows[p];
        const int matched_col = row_match[static_cast<std::size_t>(row)];
        if (matched_col == -1 ||
            (distance[static_cast<std::size_t>(matched_col)] ==
                 distance[static_cast<std::size_t>(new_col)] + 1 &&
             augmentColumnLayered(
                matched_col, adjacency,
                col_match, row_match, distance))) {
            col_match[static_cast<std::size_t>(new_col)] = row;
            row_match[static_cast<std::size_t>(row)] = new_col;
            return true;
        }
    }
    distance[static_cast<std::size_t>(new_col)] = -1;
    return false;
}

bool buildMatchingLayers(
    int n,
    const MatchingAdjacency& adjacency,
    const std::vector<int>& col_match,
    const std::vector<int>& row_match,
    std::vector<int>& distance)
{
    std::queue<int> queue;
    for (int col = 0; col < n; ++col) {
        if (col_match[static_cast<std::size_t>(col)] < 0) {
            distance[static_cast<std::size_t>(col)] = 0;
            queue.push(col);
        } else {
            distance[static_cast<std::size_t>(col)] = -1;
        }
    }
    bool found_augmenting_path = false;
    while (!queue.empty()) {
        const int col = queue.front();
        queue.pop();
        const std::vector<int>& rows = adjacency[static_cast<std::size_t>(col)];
        for (std::size_t p = 0; p < rows.size(); ++p) {
            const int row = rows[p];
            const int next_col = row_match[static_cast<std::size_t>(row)];
            if (next_col < 0) {
                found_augmenting_path = true;
            } else if (distance[static_cast<std::size_t>(next_col)] < 0) {
                distance[static_cast<std::size_t>(next_col)] =
                    distance[static_cast<std::size_t>(col)] + 1;
                queue.push(next_col);
            }
        }
    }
    return found_augmenting_path;
}

MatchingAdjacency buildMatchingAdjacency(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>* values,
    const UnsymmetricOrdering& ordering)
{
    std::vector<float> row_max(static_cast<std::size_t>(n), 0.0f);
    if (values != 0) {
        for (std::size_t p = 0; p < values->size(); ++p) {
            const int row = row_indices[p];
            row_max[static_cast<std::size_t>(row)] = std::max(
                row_max[static_cast<std::size_t>(row)], std::fabs((*values)[p]));
        }
    } else {
        std::fill(row_max.begin(), row_max.end(), 1.0f);
    }

    MatchingAdjacency adjacency(static_cast<std::size_t>(n));
    std::vector<int> seen(static_cast<std::size_t>(n), -1);
    for (int new_col = 0; new_col < n; ++new_col) {
        const int old_col = ordering.perm[static_cast<std::size_t>(new_col)];
        std::vector<std::pair<float, int> > scored_rows;
        for (int p = col_ptr[static_cast<std::size_t>(old_col)];
             p < col_ptr[static_cast<std::size_t>(old_col + 1)]; ++p) {
            const int row = row_indices[static_cast<std::size_t>(p)];
            const float magnitude = values == 0
                ? 1.0f : std::fabs((*values)[static_cast<std::size_t>(p)]);
            if (magnitude == 0.0f) {
                continue;
            }
            const float denominator = row_max[static_cast<std::size_t>(row)];
            const float score = denominator > 0.0f
                ? magnitude / denominator : magnitude;
            scored_rows.push_back(std::make_pair(score, row));
        }
        std::sort(
            scored_rows.begin(), scored_rows.end(),
            [](const std::pair<float, int>& left,
               const std::pair<float, int>& right) {
                return left.first != right.first
                    ? left.first > right.first : left.second < right.second;
            });
        std::vector<int>& rows = adjacency[static_cast<std::size_t>(new_col)];
        for (std::size_t p = 0; p < scored_rows.size(); ++p) {
            const int row = scored_rows[p].second;
            if (seen[static_cast<std::size_t>(row)] != new_col) {
                seen[static_cast<std::size_t>(row)] = new_col;
                rows.push_back(row);
            }
        }
    }
    return adjacency;
}

void computeStructuralRowPermutation(
    int n,
    const MatchingAdjacency& adjacency,
    UnsymmetricOrdering& ordering)
{
    std::vector<int> col_match(static_cast<std::size_t>(n), -1);
    std::vector<int> row_match(static_cast<std::size_t>(n), -1);
    std::vector<int> distance(static_cast<std::size_t>(n), -1);
    ordering.structural_rank = 0;

    while (buildMatchingLayers(
        n, adjacency,
        col_match, row_match, distance)) {
        for (int col = 0; col < n; ++col) {
            if (col_match[static_cast<std::size_t>(col)] < 0 &&
                augmentColumnLayered(
                    col, adjacency,
                    col_match, row_match, distance)) {
                ++ordering.structural_rank;
            }
        }
    }
    ordering.row_perm.assign(static_cast<std::size_t>(n), -1);
    ordering.row_iperm.assign(static_cast<std::size_t>(n), -1);
    std::vector<int> unused_rows;
    for (int row = 0; row < n; ++row) {
        if (row_match[static_cast<std::size_t>(row)] == -1) {
            unused_rows.push_back(row);
        }
    }
    std::size_t unused = 0;
    for (int new_row = 0; new_row < n; ++new_row) {
        int old_row = col_match[static_cast<std::size_t>(new_row)];
        if (old_row < 0) {
            old_row = unused_rows[unused++];
        }
        ordering.row_perm[static_cast<std::size_t>(new_row)] = old_row;
        ordering.row_iperm[static_cast<std::size_t>(old_row)] = new_row;
    }
}

void computeEquilibration(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values,
    UnsymmetricOrdering& ordering)
{
    ordering.row_scale.assign(static_cast<std::size_t>(n), 1.0f);
    ordering.col_scale.assign(static_cast<std::size_t>(n), 1.0f);
    std::vector<float> row_norm(static_cast<std::size_t>(n), 0.0f);
    std::vector<float> col_norm(static_cast<std::size_t>(n), 0.0f);
    for (int iteration = 0; iteration < 4; ++iteration) {
        std::fill(row_norm.begin(), row_norm.end(), 0.0f);
        for (int new_col = 0; new_col < n; ++new_col) {
            const int old_col = ordering.perm[static_cast<std::size_t>(new_col)];
            for (int p = col_ptr[static_cast<std::size_t>(old_col)];
                 p < col_ptr[static_cast<std::size_t>(old_col + 1)]; ++p) {
                const int new_row = ordering.row_iperm[static_cast<std::size_t>(
                    row_indices[static_cast<std::size_t>(p)])];
                const float scaled = std::fabs(values[static_cast<std::size_t>(p)]) *
                    ordering.row_scale[static_cast<std::size_t>(new_row)] *
                    ordering.col_scale[static_cast<std::size_t>(new_col)];
                row_norm[static_cast<std::size_t>(new_row)] = std::max(
                    row_norm[static_cast<std::size_t>(new_row)], scaled);
            }
        }
        for (int row = 0; row < n; ++row) {
            const float norm = row_norm[static_cast<std::size_t>(row)];
            if (norm > 0.0f) {
                ordering.row_scale[static_cast<std::size_t>(row)] /=
                    std::sqrt(norm);
            }
        }

        std::fill(col_norm.begin(), col_norm.end(), 0.0f);
        for (int new_col = 0; new_col < n; ++new_col) {
            const int old_col = ordering.perm[static_cast<std::size_t>(new_col)];
            for (int p = col_ptr[static_cast<std::size_t>(old_col)];
                 p < col_ptr[static_cast<std::size_t>(old_col + 1)]; ++p) {
                const int new_row = ordering.row_iperm[static_cast<std::size_t>(
                    row_indices[static_cast<std::size_t>(p)])];
                const float scaled = std::fabs(values[static_cast<std::size_t>(p)]) *
                    ordering.row_scale[static_cast<std::size_t>(new_row)] *
                    ordering.col_scale[static_cast<std::size_t>(new_col)];
                col_norm[static_cast<std::size_t>(new_col)] = std::max(
                    col_norm[static_cast<std::size_t>(new_col)], scaled);
            }
        }
        for (int col = 0; col < n; ++col) {
            const float norm = col_norm[static_cast<std::size_t>(col)];
            if (norm > 0.0f) {
                ordering.col_scale[static_cast<std::size_t>(col)] /=
                    std::sqrt(norm);
            }
        }
    }
}

} // namespace

UnsymmetricOrdering computeColamdOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values)
{
    validateCsc(n, col_ptr, row_indices, &values);
    if (row_indices.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("COLAMD int32 interface cannot represent nnz");
    }

    UnsymmetricOrdering result;
    result.perm.resize(static_cast<std::size_t>(n));
    result.iperm.resize(static_cast<std::size_t>(n));
    if (n == 0) {
        result.structural_rank = 0;
        return result;
    }

    const int nnz = static_cast<int>(row_indices.size());
    const std::size_t recommended = colamd_recommended(nnz, n, n);
    if (recommended == 0 ||
        recommended > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("COLAMD workspace size cannot be represented");
    }

    std::vector<int> workspace(recommended, 0);
    std::copy(row_indices.begin(), row_indices.end(), workspace.begin());
    std::vector<int> permutation = col_ptr;
    double knobs[COLAMD_KNOBS];
    int stats[COLAMD_STATS];
    colamd_set_defaults(knobs);
    if (!colamd(n, n, static_cast<int>(recommended), workspace.data(),
                permutation.data(), knobs, stats)) {
        throw std::runtime_error(
            "COLAMD failed with status " + std::to_string(stats[COLAMD_STATUS]));
    }

    // COLAMD returns p[new] = old. SuperLU converts this representation to
    // its old-to-new perm_c convention; this project retains both explicitly.
    std::copy(permutation.begin(), permutation.begin() + n, result.perm.begin());
    for (int new_col = 0; new_col < n; ++new_col) {
        const int old_col = result.perm[static_cast<std::size_t>(new_col)];
        if (old_col < 0 || old_col >= n) {
            throw std::runtime_error("COLAMD returned an invalid column index");
        }
        result.iperm[static_cast<std::size_t>(old_col)] = new_col;
    }
    const MatchingAdjacency adjacency = buildMatchingAdjacency(
        n, col_ptr, row_indices, &values, result);
    computeStructuralRowPermutation(n, adjacency, result);
    computeEquilibration(n, col_ptr, row_indices, values, result);
    validatePermutation(n, result);
    return result;
}

UnsymmetricOrdering computeColamdOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices)
{
    return computeColamdOrdering(
        n, col_ptr, row_indices,
        std::vector<float>(row_indices.size(), 1.0f));
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
            result.row_indices.push_back(
                ordering.row_iperm[static_cast<std::size_t>(old_row)]);
            const int new_row = result.row_indices.back();
            result.values.push_back(
                values[static_cast<std::size_t>(p)] *
                ordering.row_scale[static_cast<std::size_t>(new_row)] *
                ordering.col_scale[static_cast<std::size_t>(new_col)]);
        }
        result.col_ptr[static_cast<std::size_t>(new_col + 1)] =
            static_cast<int>(result.row_indices.size());
    }
    return result;
}
