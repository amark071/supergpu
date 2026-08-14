#include "unsymmetric_symbolic.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

using Columns = std::vector<std::vector<int> >;

void validatePattern(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices)
{
    if (n < 0 || col_ptr.size() != static_cast<std::size_t>(n + 1) ||
        col_ptr.empty() || col_ptr.front() != 0 ||
        col_ptr.back() != static_cast<int>(row_indices.size())) {
        throw std::invalid_argument("invalid CSC pattern for symbolic analysis");
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

Columns buildUpperEnvelope(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    std::size_t& edge_count)
{
    Columns upper(static_cast<std::size_t>(n));
    for (int col = 0; col < n; ++col) {
        for (int p = col_ptr[static_cast<std::size_t>(col)];
             p < col_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
            const int row = row_indices[static_cast<std::size_t>(p)];
            if (row != col) {
                upper[static_cast<std::size_t>(std::max(row, col))].push_back(
                    std::min(row, col));
            }
        }
    }
    edge_count = 0;
    for (int col = 0; col < n; ++col) {
        std::vector<int>& rows = upper[static_cast<std::size_t>(col)];
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        edge_count += rows.size();
    }
    return upper;
}

std::vector<int> buildEliminationTree(int n, const Columns& upper)
{
    std::vector<int> parent(static_cast<std::size_t>(n), -1);
    std::vector<int> ancestor(static_cast<std::size_t>(n), -1);
    for (int col = 0; col < n; ++col) {
        const std::vector<int>& rows = upper[static_cast<std::size_t>(col)];
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
    return parent;
}

Columns buildFactorColumns(
    int n,
    const Columns& upper,
    const std::vector<int>& parent)
{
    Columns factor(static_cast<std::size_t>(n));
    for (int col = 0; col < n; ++col) {
        factor[static_cast<std::size_t>(col)].push_back(col);
    }
    std::vector<int> marked(static_cast<std::size_t>(n), -1);
    for (int high = 0; high < n; ++high) {
        const std::vector<int>& lows = upper[static_cast<std::size_t>(high)];
        for (std::size_t p = 0; p < lows.size(); ++p) {
            int low = lows[p];
            while (low != -1 && low < high &&
                   marked[static_cast<std::size_t>(low)] != high) {
                marked[static_cast<std::size_t>(low)] = high;
                factor[static_cast<std::size_t>(low)].push_back(high);
                low = parent[static_cast<std::size_t>(low)];
            }
        }
    }
    for (int col = 0; col < n; ++col) {
        std::vector<int>& rows = factor[static_cast<std::size_t>(col)];
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }
    return factor;
}

bool relaxedMergeAllowed(int width, int zeros, int entries)
{
    if (width <= 4) {
        return true;
    }
    const double ratio = entries == 0
        ? 0.0 : static_cast<double>(zeros) / static_cast<double>(entries);
    if (width <= 16) {
        return ratio <= 0.8;
    }
    if (width <= 48) {
        return ratio <= 0.1;
    }
    return width <= 128 && ratio <= 0.05;
}

std::vector<int> formRelaxedSupernodes(
    int n,
    const std::vector<int>& parent,
    const Columns& factor)
{
    std::vector<int> ptr;
    ptr.push_back(0);
    int begin = 0;
    while (begin < n) {
        int end = begin + 1;
        std::vector<int> envelope = factor[static_cast<std::size_t>(begin)];
        int envelope_size = static_cast<int>(envelope.size());
        int actual_entries = envelope_size;
        while (end < n && parent[static_cast<std::size_t>(end - 1)] == end) {
            std::vector<int> merged;
            std::set_union(
                envelope.begin(), envelope.end(),
                factor[static_cast<std::size_t>(end)].begin(),
                factor[static_cast<std::size_t>(end)].end(),
                std::back_inserter(merged));
            envelope_size = static_cast<int>(merged.size());
            actual_entries += static_cast<int>(factor[static_cast<std::size_t>(end)].size());
            const int width = end - begin + 1;
            const int dense_entries = width * envelope_size;
            if (!relaxedMergeAllowed(width, dense_entries - actual_entries,
                                     dense_entries)) {
                break;
            }
            envelope.swap(merged);
            ++end;
        }
        ptr.push_back(end);
        begin = end;
    }
    return ptr;
}

void buildFronts(
    int n,
    const Columns& factor,
    UnsymmetricSymbolicResult& result)
{
    const int supernodes = static_cast<int>(result.supernode_ptr.size()) - 1;
    std::vector<int> column_to_supernode(static_cast<std::size_t>(n), -1);
    result.front_ptr.push_back(0);
    for (int node = 0; node < supernodes; ++node) {
        const int begin = result.supernode_ptr[static_cast<std::size_t>(node)];
        const int end = result.supernode_ptr[static_cast<std::size_t>(node + 1)];
        for (int col = begin; col < end; ++col) {
            column_to_supernode[static_cast<std::size_t>(col)] = node;
            result.front_indices.push_back(col);
        }
        std::vector<int> update;
        for (int col = begin; col < end; ++col) {
            const std::vector<int>& rows = factor[static_cast<std::size_t>(col)];
            update.insert(update.end(), rows.begin(), rows.end());
        }
        std::sort(update.begin(), update.end());
        update.erase(std::unique(update.begin(), update.end()), update.end());
        for (std::size_t p = 0; p < update.size(); ++p) {
            if (update[p] >= end) {
                result.front_indices.push_back(update[p]);
            }
        }
        result.front_ptr.push_back(static_cast<int>(result.front_indices.size()));
    }

    result.supernode_parent.assign(static_cast<std::size_t>(supernodes), -1);
    for (int node = 0; node < supernodes; ++node) {
        const int last = result.supernode_ptr[static_cast<std::size_t>(node + 1)] - 1;
        const int parent_col = result.column_parent[static_cast<std::size_t>(last)];
        if (parent_col >= 0) {
            result.supernode_parent[static_cast<std::size_t>(node)] =
                column_to_supernode[static_cast<std::size_t>(parent_col)];
        }
    }
}

} // namespace

UnsymmetricSymbolicResult analyzeUnsymmetricSupernodes(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices)
{
    validatePattern(n, col_ptr, row_indices);
    UnsymmetricSymbolicResult result;
    if (n == 0) {
        result.supernode_ptr.push_back(0);
        result.front_ptr.push_back(0);
        return result;
    }

    const Columns upper = buildUpperEnvelope(
        n, col_ptr, row_indices, result.envelope_off_diagonal_nonzeros);
    result.column_parent = buildEliminationTree(n, upper);
    const Columns factor = buildFactorColumns(n, upper, result.column_parent);
    result.column_count.resize(static_cast<std::size_t>(n));
    result.symbolic_factor_nonzeros = 0;
    for (int col = 0; col < n; ++col) {
        result.column_count[static_cast<std::size_t>(col)] =
            static_cast<int>(factor[static_cast<std::size_t>(col)].size());
        result.symbolic_factor_nonzeros +=
            factor[static_cast<std::size_t>(col)].size();
    }
    result.supernode_ptr = formRelaxedSupernodes(
        n, result.column_parent, factor);
    buildFronts(n, factor, result);
    return result;
}
