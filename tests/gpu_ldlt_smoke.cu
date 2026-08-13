#include "gpu_supernodal_ldlt.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

CholmodSymbolicResult oneFrontSymbolic(int n)
{
    CholmodSymbolicResult symbolic;
    symbolic.column_parent.assign(static_cast<std::size_t>(n), -1);
    symbolic.column_count.assign(static_cast<std::size_t>(n), n);
    symbolic.supernode_ptr.push_back(0);
    symbolic.supernode_ptr.push_back(n);
    symbolic.supernode_parent.push_back(-1);
    symbolic.row_ptr.push_back(0);
    symbolic.row_ptr.push_back(n);
    for (int row = 0; row < n; ++row) {
        symbolic.supernode_rows.push_back(row);
    }
    return symbolic;
}

float residualInfinityNorm(
    const std::vector<float>& dense,
    const std::vector<float>& x,
    const std::vector<float>& rhs)
{
    const int n = static_cast<int>(x.size());
    float maximum = 0.0f;
    for (int row = 0; row < n; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < n; ++col) {
            sum += dense[row + col * n] * x[col];
        }
        maximum = std::max(maximum, std::fabs(sum - rhs[row]));
    }
    return maximum;
}

bool runCase(
    const std::vector<float>& dense,
    const std::vector<int>& col_ptr,
    const std::vector<int>& rows,
    const CholmodSymbolicResult& symbolic,
    bool require_two_by_two,
    bool require_large,
    const char* name)
{
    const int n = static_cast<int>(symbolic.column_parent.size());
    std::vector<float> values;
    for (int col = 0; col < n; ++col) {
        for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
            values.push_back(dense[rows[p] + col * n]);
        }
    }

    GpuSupernodalLdltFactor factor;
    const GpuLdltStatistics statistics = factor.factorize(
        n, col_ptr, rows, values, symbolic);
    if (!factor.complete() ||
        !statistics.sorted_csc_fast_path ||
        (require_two_by_two && statistics.accepted_two_by_two_pivots == 0) ||
        (require_large && statistics.large_panel_nodes == 0)) {
        std::cerr << name << ": " << factor.diagnostic() << '\n';
        return false;
    }

    std::vector<float> rhs(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        rhs[i] = 0.25f * static_cast<float>((i % 7) - 3);
    }
    const std::vector<float> solution = factor.solve(rhs);
    const float residual = residualInfinityNorm(dense, solution, rhs);
    std::cout << name << " FP32 residual infinity norm = " << residual << '\n';
    return residual <= 2.0e-4f;
}

} // namespace

int main()
{
    // A zero diagonal forces the first 2x2 Bunch-Kaufman pivot.
    const int n = 4;
    const std::vector<float> dense = {
         0.0f, 2.0f, 0.0f, 0.0f,
         2.0f, 0.0f, 1.0f, 0.0f,
         0.0f, 1.0f, 3.0f, 1.0f,
         0.0f, 0.0f, 1.0f, 2.0f
    };
    const std::vector<int> col_ptr = {0, 2, 5, 8, 10};
    const std::vector<int> rows = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    if (!runCase(dense, col_ptr, rows, oneFrontSymbolic(n), true, false,
                 "medium BK")) {
        return 1;
    }

    // A repeated diagonal entry forces the compatibility hash path. The two
    // copies sum to the same diagonal represented by fallback_dense.
    const std::vector<float> fallback_dense = {
        4.0f, 1.0f,
        1.0f, 3.0f
    };
    const std::vector<int> fallback_col_ptr = {0, 3, 5};
    const std::vector<int> fallback_rows = {0, 0, 1, 0, 1};
    const std::vector<float> fallback_values = {2.0f, 2.0f, 1.0f, 1.0f, 3.0f};
    GpuSupernodalLdltFactor fallback_factor;
    const GpuLdltStatistics fallback_statistics = fallback_factor.factorize(
        2, fallback_col_ptr, fallback_rows, fallback_values,
        oneFrontSymbolic(2));
    const std::vector<float> fallback_rhs = {1.0f, -0.5f};
    const std::vector<float> fallback_solution =
        fallback_factor.solve(fallback_rhs);
    const float fallback_residual = residualInfinityNorm(
        fallback_dense, fallback_solution, fallback_rhs);
    std::cout << "duplicate CSC fallback FP32 residual infinity norm = "
              << fallback_residual << '\n';
    if (!fallback_factor.complete() ||
        fallback_statistics.sorted_csc_fast_path ||
        fallback_residual > 2.0e-4f) {
        return 2;
    }

    // Column 0 is delayed and forms a stable 2x2 pivot at its parent.
    const std::vector<float> delayed_dense = {
        0.0f, 1.0f,
        1.0f, 0.0f
    };
    const std::vector<int> delayed_col_ptr = {0, 2, 4};
    const std::vector<int> delayed_rows = {0, 1, 0, 1};
    CholmodSymbolicResult delayed_symbolic;
    delayed_symbolic.column_parent = {1, -1};
    delayed_symbolic.column_count = {2, 1};
    delayed_symbolic.supernode_ptr = {0, 1, 2};
    delayed_symbolic.supernode_parent = {1, -1};
    delayed_symbolic.row_ptr = {0, 2, 3};
    delayed_symbolic.supernode_rows = {0, 1, 1};
    if (!runCase(delayed_dense, delayed_col_ptr, delayed_rows,
                 delayed_symbolic, true, false, "delayed child-to-parent")) {
        return 3;
    }

    // A nonzero diagonal is not sufficient for a stable single-column pivot.
    // The coupling to the update row dominates, so column 0 must be delayed and
    // paired with column 1 at the parent instead of producing an O(1e8) multiplier.
    const std::vector<float> unstable_single_dense = {
        1.0e-8f, 1.0f,
        1.0f,    0.0f
    };
    if (!runCase(unstable_single_dense, delayed_col_ptr, delayed_rows,
                 delayed_symbolic, true, false, "unstable single delayed")) {
        return 4;
    }

    const std::vector<float> chain_dense = {
        4.0f, 1.0f, 0.0f,
        1.0f, 4.0f, 1.0f,
        0.0f, 1.0f, 4.0f
    };
    const std::vector<int> chain_col_ptr = {0, 2, 5, 7};
    const std::vector<int> chain_rows = {0, 1, 0, 1, 2, 1, 2};
    CholmodSymbolicResult chain_symbolic;
    chain_symbolic.column_parent = {1, 2, -1};
    chain_symbolic.column_count = {2, 2, 1};
    chain_symbolic.supernode_ptr = {0, 1, 2, 3};
    chain_symbolic.supernode_parent = {1, 2, -1};
    chain_symbolic.row_ptr = {0, 2, 4, 5};
    chain_symbolic.supernode_rows = {0, 1, 1, 2, 2};
    if (!runCase(chain_dense, chain_col_ptr, chain_rows,
                  chain_symbolic, false, false, "three-level contribution")) {
        return 5;
    }

    // Width 65 forces the large-panel path.
    const int large_n = 65;
    std::vector<float> large_dense(
        static_cast<std::size_t>(large_n) * large_n, 0.0f);
    large_dense[1] = 2.0f;
    large_dense[large_n] = 2.0f;
    for (int i = 2; i < large_n; ++i) {
        large_dense[i + i * large_n] = 2.0f + 0.01f * static_cast<float>(i);
    }
    for (int i = 1; i + 1 < large_n; ++i) {
        large_dense[i + 1 + i * large_n] = 0.05f;
        large_dense[i + (i + 1) * large_n] = 0.05f;
    }
    std::vector<int> large_col_ptr(static_cast<std::size_t>(large_n + 1), 0);
    std::vector<int> large_rows;
    for (int col = 0; col < large_n; ++col) {
        for (int row = 0; row < large_n; ++row) {
            if (large_dense[row + col * large_n] != 0.0f) {
                large_rows.push_back(row);
            }
        }
        large_col_ptr[col + 1] = static_cast<int>(large_rows.size());
    }
    if (!runCase(large_dense, large_col_ptr, large_rows,
                  oneFrontSymbolic(large_n), true, true, "large panel BK")) {
        return 6;
    }

    // An unstable column inside a large panel is delayed to its parent.
    const int delayed_large_n = 66;
    std::vector<float> delayed_large_dense(
        static_cast<std::size_t>(delayed_large_n) * delayed_large_n, 0.0f);
    for (int i = 0; i < 65; ++i) {
        if (i != 10) {
            delayed_large_dense[i + i * delayed_large_n] =
                2.0f + 0.01f * static_cast<float>(i);
        }
    }
    delayed_large_dense[65 + 10 * delayed_large_n] = 1.0f;
    delayed_large_dense[10 + 65 * delayed_large_n] = 1.0f;
    std::vector<int> delayed_large_col_ptr(
        static_cast<std::size_t>(delayed_large_n + 1), 0);
    std::vector<int> delayed_large_rows;
    for (int col = 0; col < delayed_large_n; ++col) {
        for (int row = 0; row < delayed_large_n; ++row) {
            if (delayed_large_dense[row + col * delayed_large_n] != 0.0f) {
                delayed_large_rows.push_back(row);
            }
        }
        delayed_large_col_ptr[col + 1] =
            static_cast<int>(delayed_large_rows.size());
    }
    CholmodSymbolicResult delayed_large_symbolic;
    delayed_large_symbolic.column_parent.assign(delayed_large_n, -1);
    delayed_large_symbolic.column_parent[64] = 65;
    delayed_large_symbolic.column_count.assign(delayed_large_n, 1);
    delayed_large_symbolic.supernode_ptr = {0, 65, 66};
    delayed_large_symbolic.supernode_parent = {1, -1};
    delayed_large_symbolic.row_ptr = {0, 66, 67};
    for (int row = 0; row < delayed_large_n; ++row) {
        delayed_large_symbolic.supernode_rows.push_back(row);
    }
    delayed_large_symbolic.supernode_rows.push_back(65);
    if (!runCase(delayed_large_dense, delayed_large_col_ptr, delayed_large_rows,
                  delayed_large_symbolic, true, true, "large panel delay")) {
        return 7;
    }
    return 0;
}
