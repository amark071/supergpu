#include "gpu_supernodal_lu.hpp"
#include "unsymmetric_ordering.hpp"
#include "unsymmetric_refinement.hpp"
#include "unsymmetric_symbolic.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

UnsymmetricSymbolicResult oneFrontSymbolic(int n)
{
    UnsymmetricSymbolicResult symbolic;
    symbolic.column_parent.assign(static_cast<std::size_t>(n), -1);
    symbolic.column_count.assign(static_cast<std::size_t>(n), n);
    symbolic.supernode_ptr = {0, n};
    symbolic.supernode_parent = {-1};
    symbolic.front_ptr = {0, n};
    for (int i = 0; i < n; ++i) {
        symbolic.front_indices.push_back(i);
    }
    return symbolic;
}

void denseToCsc(
    int n,
    const std::vector<float>& dense,
    std::vector<int>& col_ptr,
    std::vector<int>& rows,
    std::vector<float>& values)
{
    col_ptr.assign(static_cast<std::size_t>(n + 1), 0);
    rows.clear();
    values.clear();
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            const float value = dense[static_cast<std::size_t>(row + col * n)];
            if (value != 0.0f) {
                rows.push_back(row);
                values.push_back(value);
            }
        }
        col_ptr[static_cast<std::size_t>(col + 1)] =
            static_cast<int>(rows.size());
    }
}

float residualInfinity(
    const std::vector<float>& dense,
    const std::vector<float>& solution,
    const std::vector<float>& rhs)
{
    const int n = static_cast<int>(solution.size());
    float result = 0.0f;
    for (int row = 0; row < n; ++row) {
        float value = 0.0f;
        for (int col = 0; col < n; ++col) {
            value += dense[static_cast<std::size_t>(row + col * n)] *
                solution[static_cast<std::size_t>(col)];
        }
        result = std::max(result, std::fabs(value - rhs[row]));
    }
    return result;
}

} // namespace

int main()
{
    // The first column requires a row swap; the matrix is intentionally not
    // symmetric. Values are column-major.
    const int n = 4;
    const std::vector<float> dense = {
        0.0f, 3.0f, 0.0f, 0.0f,
        2.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 4.0f, 5.0f, 0.0f,
        1.0f, 0.0f, 2.0f, 6.0f
    };
    std::vector<int> col_ptr;
    std::vector<int> rows;
    std::vector<float> values;
    denseToCsc(n, dense, col_ptr, rows, values);

    const UnsymmetricSymbolicResult symbolic = analyzeUnsymmetricSupernodes(
        n, col_ptr, rows);
    GpuSupernodalLuFactor factor;
    const GpuLuStatistics statistics = factor.factorize(
        n, col_ptr, rows, values, symbolic);
    if (!factor.complete() || statistics.accepted_pivots != 4) {
        std::cerr << factor.diagnostic() << '\n';
        return 1;
    }
    const std::vector<float> rhs = {1.0f, -2.0f, 0.5f, 3.0f};
    const std::vector<float> solution = factor.solve(rhs);
    const float residual = residualInfinity(dense, solution, rhs);
    std::cout << "general LU FP32 residual infinity norm = "
              << residual << '\n';
    if (residual > 2.0e-4f) {
        return 2;
    }

    // In node 0, column 0 is unusable because its only nonzero is in the
    // update row. Column 1 is pulled in from the right, column 0 is delayed,
    // and the root later accepts it.
    const int delayed_n = 3;
    const std::vector<float> delayed_dense = {
        0.0f, 0.0f, 1.0f,
        0.0f, 2.0f, 0.0f,
        3.0f, 0.0f, 0.0f
    };
    const std::vector<int> delayed_col_ptr = {0, 1, 2, 3};
    const std::vector<int> delayed_rows = {2, 1, 0};
    const std::vector<float> delayed_values = {1.0f, 2.0f, 3.0f};
    UnsymmetricSymbolicResult delayed_symbolic;
    delayed_symbolic.column_parent = {1, 2, -1};
    delayed_symbolic.column_count = {3, 2, 1};
    delayed_symbolic.supernode_ptr = {0, 2, 3};
    delayed_symbolic.supernode_parent = {1, -1};
    delayed_symbolic.front_ptr = {0, 3, 4};
    delayed_symbolic.front_indices = {0, 1, 2, 2};

    GpuSupernodalLuFactor delayed_factor;
    const GpuLuStatistics delayed_statistics = delayed_factor.factorize(
        delayed_n, delayed_col_ptr, delayed_rows,
        delayed_values, delayed_symbolic);
    const std::vector<float> delayed_rhs = {2.0f, -1.0f, 4.0f};
    if (!delayed_factor.complete() || delayed_statistics.delayed_columns == 0) {
        std::cerr << "delayed replacement: "
                  << delayed_factor.diagnostic() << '\n';
        return 3;
    }
    const std::vector<float> delayed_solution =
        delayed_factor.solve(delayed_rhs);
    const float delayed_residual = residualInfinity(
        delayed_dense, delayed_solution, delayed_rhs);
    std::cout << "delayed-column LU FP32 residual infinity norm = "
              << delayed_residual << '\n';
    if (delayed_residual > 2.0e-4f) {
        return 4;
    }

    // A banded general matrix exercises multiple automatically generated
    // fronts and child contribution propagation.
    const int band_n = 12;
    std::vector<float> band_dense(
        static_cast<std::size_t>(band_n * band_n), 0.0f);
    for (int i = 0; i < band_n; ++i) {
        band_dense[static_cast<std::size_t>(i + i * band_n)] =
            5.0f + 0.1f * i;
        if (i + 1 < band_n) {
            band_dense[static_cast<std::size_t>(i + 1 + i * band_n)] = -1.0f;
            band_dense[static_cast<std::size_t>(i + (i + 1) * band_n)] = 0.4f;
        }
        if (i + 2 < band_n) {
            band_dense[static_cast<std::size_t>(i + 2 + i * band_n)] = 0.2f;
        }
    }
    std::vector<int> band_col_ptr;
    std::vector<int> band_rows;
    std::vector<float> band_values;
    denseToCsc(
        band_n, band_dense, band_col_ptr, band_rows, band_values);
    const UnsymmetricOrdering band_ordering = computeMatchingAmdOrdering(
        band_n, band_col_ptr, band_rows, band_values);
    const UnsymmetricPermutedCsc band_ordered = applyUnsymmetricPermutationCsc(
        band_n, band_col_ptr, band_rows, band_values, band_ordering);
    const UnsymmetricSymbolicResult band_symbolic = analyzeUnsymmetricSupernodes(
        band_n, band_ordered.col_ptr, band_ordered.row_indices);
    GpuSupernodalLuFactor band_factor;
    band_factor.factorize(
        band_n, band_ordered.col_ptr, band_ordered.row_indices,
        band_ordered.values, band_symbolic);
    std::vector<float> band_rhs(static_cast<std::size_t>(band_n), 0.0f);
    for (int col = 0; col < band_n; ++col) {
        for (int p = band_ordered.col_ptr[static_cast<std::size_t>(col)];
             p < band_ordered.col_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
            band_rhs[static_cast<std::size_t>(band_ordered.row_indices[p])] +=
                band_ordered.values[p];
        }
    }
    const std::vector<float> band_solution = band_factor.solve(band_rhs);
    const std::vector<float> band_expected(static_cast<std::size_t>(band_n), 1.0f);
    const float band_residual = residualInfinity(
        [&]() {
            std::vector<float> ordered_dense(
                static_cast<std::size_t>(band_n * band_n), 0.0f);
            for (int col = 0; col < band_n; ++col) {
                for (int p = band_ordered.col_ptr[col];
                     p < band_ordered.col_ptr[col + 1]; ++p) {
                    ordered_dense[static_cast<std::size_t>(
                        band_ordered.row_indices[p] + col * band_n)] +=
                        band_ordered.values[p];
                }
            }
            return ordered_dense;
        }(), band_solution, band_rhs);
    std::cout << "automatic multifrontal LU FP32 residual infinity norm = "
              << band_residual << '\n';
    if (!band_factor.complete() || band_residual > 5.0e-4f ||
        band_solution.size() != band_expected.size()) {
        return 5;
    }
    std::vector<float> band_original_rhs(
        static_cast<std::size_t>(band_n), 0.0f);
    for (int row = 0; row < band_n; ++row) {
        for (int col = 0; col < band_n; ++col) {
            band_original_rhs[static_cast<std::size_t>(row)] +=
                band_dense[static_cast<std::size_t>(row + col * band_n)];
        }
    }
    const UnsymmetricRefinementResult band_refined =
        solveGeneralWithIterativeRefinement(
            band_n, band_col_ptr, band_rows, band_values,
            band_original_rhs, band_ordering, band_factor);
    const float refined_residual = residualInfinity(
        band_dense, band_refined.solution, band_original_rhs);
    std::cout << "iteratively refined LU FP32 residual infinity norm = "
              << refined_residual << '\n';
    if (refined_residual > 5.0e-4f) {
        return 9;
    }

    // Width 65 selects the cuBLAS TRSM/GEMM panel path.
    const int large_n = 65;
    std::vector<float> large_dense(
        static_cast<std::size_t>(large_n * large_n), 0.0f);
    for (int col = 0; col < large_n; ++col) {
        large_dense[static_cast<std::size_t>(col + col * large_n)] = 8.0f;
        for (int row = 0; row < large_n; ++row) {
            if (row != col) {
                large_dense[static_cast<std::size_t>(row + col * large_n)] =
                    0.002f * static_cast<float>(1 + (row + 3 * col) % 11);
            }
        }
    }
    std::vector<int> large_col_ptr;
    std::vector<int> large_rows;
    std::vector<float> large_values;
    denseToCsc(
        large_n, large_dense, large_col_ptr, large_rows, large_values);
    GpuSupernodalLuFactor large_factor;
    const GpuLuStatistics large_statistics = large_factor.factorize(
        large_n, large_col_ptr, large_rows, large_values,
        oneFrontSymbolic(large_n));
    std::vector<float> large_rhs(static_cast<std::size_t>(large_n), 0.0f);
    for (int row = 0; row < large_n; ++row) {
        for (int col = 0; col < large_n; ++col) {
            large_rhs[static_cast<std::size_t>(row)] +=
                large_dense[static_cast<std::size_t>(row + col * large_n)];
        }
    }
    const std::vector<float> large_solution = large_factor.solve(large_rhs);
    const float large_residual = residualInfinity(
        large_dense, large_solution, large_rhs);
    std::cout << "large panel LU FP32 residual infinity norm = "
              << large_residual << '\n';
    if (!large_factor.complete() || large_statistics.large_front_nodes != 1 ||
        large_residual > 2.0e-3f) {
        return 6;
    }

    // An unusable column in the middle of a 65-column front forces a partial
    // panel flush, right-to-left replacement, and propagation to the root.
    const int delayed_large_n = 66;
    std::vector<float> delayed_large_dense(
        static_cast<std::size_t>(delayed_large_n * delayed_large_n), 0.0f);
    for (int i = 0; i < 65; ++i) {
        if (i != 10) {
            delayed_large_dense[static_cast<std::size_t>(
                i + i * delayed_large_n)] = 2.0f + 0.01f * i;
        }
    }
    delayed_large_dense[static_cast<std::size_t>(65 + 10 * delayed_large_n)] =
        1.0f;
    delayed_large_dense[static_cast<std::size_t>(10 + 65 * delayed_large_n)] =
        1.0f;
    std::vector<int> delayed_large_col_ptr;
    std::vector<int> delayed_large_rows;
    std::vector<float> delayed_large_values;
    denseToCsc(
        delayed_large_n, delayed_large_dense, delayed_large_col_ptr,
        delayed_large_rows, delayed_large_values);
    UnsymmetricSymbolicResult delayed_large_symbolic;
    delayed_large_symbolic.column_parent.assign(delayed_large_n, -1);
    delayed_large_symbolic.column_parent[64] = 65;
    delayed_large_symbolic.column_count.assign(delayed_large_n, 1);
    delayed_large_symbolic.supernode_ptr = {0, 65, 66};
    delayed_large_symbolic.supernode_parent = {1, -1};
    delayed_large_symbolic.front_ptr = {0, 66, 67};
    for (int id = 0; id < delayed_large_n; ++id) {
        delayed_large_symbolic.front_indices.push_back(id);
    }
    delayed_large_symbolic.front_indices.push_back(65);
    GpuSupernodalLuFactor delayed_large_factor;
    const GpuLuStatistics delayed_large_statistics =
        delayed_large_factor.factorize(
            delayed_large_n, delayed_large_col_ptr, delayed_large_rows,
            delayed_large_values, delayed_large_symbolic);
    std::vector<float> delayed_large_rhs(
        static_cast<std::size_t>(delayed_large_n), 0.0f);
    for (int row = 0; row < delayed_large_n; ++row) {
        for (int col = 0; col < delayed_large_n; ++col) {
            delayed_large_rhs[static_cast<std::size_t>(row)] +=
                delayed_large_dense[static_cast<std::size_t>(
                    row + col * delayed_large_n)];
        }
    }
    const std::vector<float> delayed_large_solution =
        delayed_large_factor.solve(delayed_large_rhs);
    const float delayed_large_residual = residualInfinity(
        delayed_large_dense, delayed_large_solution, delayed_large_rhs);
    std::cout << "large delayed-column LU FP32 residual infinity norm = "
              << delayed_large_residual << '\n';
    if (!delayed_large_factor.complete() ||
        delayed_large_statistics.delayed_columns == 0 ||
        delayed_large_residual > 2.0e-3f) {
        return 7;
    }

    // Two dense three-column fronts verify a nontrivial Schur complement.
    const int block_n = 6;
    std::vector<float> block_dense(
        static_cast<std::size_t>(block_n * block_n), 0.0f);
    for (int col = 0; col < block_n; ++col) {
        for (int row = 0; row < block_n; ++row) {
            block_dense[static_cast<std::size_t>(row + col * block_n)] =
                row == col
                    ? 7.0f + 0.1f * col
                    : 0.03f * static_cast<float>(1 + (2 * row + col) % 7);
        }
    }
    std::vector<int> block_col_ptr;
    std::vector<int> block_rows;
    std::vector<float> block_values;
    denseToCsc(
        block_n, block_dense, block_col_ptr, block_rows, block_values);
    UnsymmetricSymbolicResult block_symbolic;
    block_symbolic.column_parent = {1, 2, 3, 4, 5, -1};
    block_symbolic.column_count = {6, 5, 4, 3, 2, 1};
    block_symbolic.supernode_ptr = {0, 3, 6};
    block_symbolic.supernode_parent = {1, -1};
    block_symbolic.front_ptr = {0, 6, 9};
    block_symbolic.front_indices = {0, 1, 2, 3, 4, 5, 3, 4, 5};
    GpuSupernodalLuFactor block_factor;
    block_factor.factorize(
        block_n, block_col_ptr, block_rows, block_values, block_symbolic);
    std::vector<float> block_rhs(static_cast<std::size_t>(block_n), 0.0f);
    for (int row = 0; row < block_n; ++row) {
        for (int col = 0; col < block_n; ++col) {
            block_rhs[static_cast<std::size_t>(row)] +=
                block_dense[static_cast<std::size_t>(row + col * block_n)];
        }
    }
    const std::vector<float> block_solution = block_factor.solve(block_rhs);
    const float block_residual = residualInfinity(
        block_dense, block_solution, block_rhs);
    std::cout << "two-front Schur LU FP32 residual infinity norm = "
              << block_residual << '\n';
    if (!block_factor.complete() || block_residual > 2.0e-4f) {
        return 8;
    }
    return 0;
}
