#include "unsymmetric_ordering.hpp"
#include "unsymmetric_symbolic.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

int main()
{
    // A perfect structural matching exists, but none of these five columns is
    // required to start with its matching row on the diagonal.
    const int n = 5;
    const std::vector<int> col_ptr = {0, 1, 3, 4, 5, 6};
    const std::vector<int> rows = {2, 0, 2, 4, 1, 3};
    const std::vector<float> values(rows.size(), 1.0f);

    const UnsymmetricOrdering ordering = computeColamdOrdering(
        n, col_ptr, rows, values);
    if (ordering.structural_rank != n) {
        std::cerr << "unexpected structural rank\n";
        return 1;
    }
    const UnsymmetricPermutedCsc ordered = applyUnsymmetricPermutationCsc(
        n, col_ptr, rows, values, ordering);
    for (int col = 0; col < n; ++col) {
        bool diagonal = false;
        for (int p = ordered.col_ptr[static_cast<std::size_t>(col)];
             p < ordered.col_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
            diagonal = diagonal ||
                ordered.row_indices[static_cast<std::size_t>(p)] == col;
        }
        if (!diagonal) {
            std::cerr << "structural matching did not form a diagonal\n";
            return 2;
        }
    }

    const UnsymmetricSymbolicResult symbolic = analyzeUnsymmetricSupernodes(
        n, ordered.col_ptr, ordered.row_indices);
    std::vector<int> owner(static_cast<std::size_t>(n), -1);
    for (std::size_t node = 0; node < symbolic.supernode_parent.size(); ++node) {
        for (int col = symbolic.supernode_ptr[node];
             col < symbolic.supernode_ptr[node + 1]; ++col) {
            owner[static_cast<std::size_t>(col)] = static_cast<int>(node);
        }
    }
    for (int col = 0; col < n; ++col) {
        for (int p = ordered.col_ptr[static_cast<std::size_t>(col)];
             p < ordered.col_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
            const int row = ordered.row_indices[static_cast<std::size_t>(p)];
            const int node = owner[static_cast<std::size_t>(std::min(row, col))];
            const std::vector<int>::const_iterator begin =
                symbolic.front_indices.begin() + symbolic.front_ptr[node];
            const std::vector<int>::const_iterator end =
                symbolic.front_indices.begin() + symbolic.front_ptr[node + 1];
            if (std::find(begin, end, row) == end ||
                std::find(begin, end, col) == end) {
                std::cerr << "symbolic envelope omitted an input entry\n";
                return 3;
            }
        }
    }
    std::cout << "COLAMD, matching, and symbolic envelope smoke test passed\n";
    return 0;
}
