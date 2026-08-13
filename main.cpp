#include "cholmod_symbolic.hpp"
#include "gpu_supernodal_ldlt.hpp"
#include "io_func.hpp"
#include "ordering.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef SUPERNODAL_VISUALIZATION_DIR
#define SUPERNODAL_VISUALIZATION_DIR "."
#endif

namespace {

struct SymmetricEntryAccumulator {
    SymmetricEntryAccumulator()
        : lower(0.0f), upper(0.0f), lower_count(0), upper_count(0)
    {
    }

    float lower;
    float upper;
    int lower_count;
    int upper_count;
};

struct ResidualVerification {
    float matrix_infinity_norm;
    float rhs_infinity_norm;
    float solution_infinity_norm;
    float absolute_residual;
    float relative_residual;
    float scaled_backward_error;
    float relative_solution_error;
};

std::uint64_t symmetricKey(int row, int col)
{
    const std::uint32_t high =
        static_cast<std::uint32_t>(std::max(row, col));
    const std::uint32_t low =
        static_cast<std::uint32_t>(std::min(row, col));
    return (static_cast<std::uint64_t>(high) << 32) | low;
}

std::unordered_map<std::uint64_t, SymmetricEntryAccumulator>
buildSymmetricEntries(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values)
{
    std::unordered_map<std::uint64_t, SymmetricEntryAccumulator> entries;
    entries.reserve(values.size());
    for (int col = 0; col < n; ++col) {
        for (int p = col_ptr[static_cast<std::size_t>(col)];
             p < col_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
            const int row = row_indices[static_cast<std::size_t>(p)];
            SymmetricEntryAccumulator& accumulator =
                entries[symmetricKey(row, col)];
            if (row >= col) {
                accumulator.lower += values[static_cast<std::size_t>(p)];
                ++accumulator.lower_count;
            } else {
                accumulator.upper += values[static_cast<std::size_t>(p)];
                ++accumulator.upper_count;
            }
        }
    }
    return entries;
}

float symmetricValue(
    int row,
    int col,
    const SymmetricEntryAccumulator& accumulator,
    bool symmetrize_input)
{
    if (row == col) {
        return accumulator.lower + accumulator.upper;
    }
    if (accumulator.lower_count != 0 && accumulator.upper_count != 0) {
        return symmetrize_input
            ? 0.5f * (accumulator.lower + accumulator.upper)
            : accumulator.lower;
    }
    return accumulator.lower_count != 0
        ? accumulator.lower : accumulator.upper;
}

std::vector<float> symmetricMatrixVectorProduct(
    int n,
    const std::unordered_map<std::uint64_t, SymmetricEntryAccumulator>& entries,
    bool symmetrize_input,
    const std::vector<float>& x,
    float* matrix_infinity_norm)
{
    std::vector<float> result(static_cast<std::size_t>(n), 0.0f);
    std::vector<float> absolute_row_sums(static_cast<std::size_t>(n), 0.0f);

    for (std::unordered_map<std::uint64_t, SymmetricEntryAccumulator>::const_iterator
             iterator = entries.begin(); iterator != entries.end(); ++iterator) {
        const int row = static_cast<int>(iterator->first >> 32);
        const int col = static_cast<int>(iterator->first & 0xffffffffu);
        const float value = symmetricValue(
            row, col, iterator->second, symmetrize_input);
        result[static_cast<std::size_t>(row)] +=
            value * x[static_cast<std::size_t>(col)];
        absolute_row_sums[static_cast<std::size_t>(row)] += std::fabs(value);
        if (row != col) {
            result[static_cast<std::size_t>(col)] +=
                value * x[static_cast<std::size_t>(row)];
            absolute_row_sums[static_cast<std::size_t>(col)] += std::fabs(value);
        }
    }

    if (matrix_infinity_norm != 0) {
        *matrix_infinity_norm = *std::max_element(
            absolute_row_sums.begin(), absolute_row_sums.end());
    }
    return result;
}

float infinityNorm(const std::vector<float>& values)
{
    float norm = 0.0f;
    for (std::size_t i = 0; i < values.size(); ++i) {
        norm = std::max(norm, std::fabs(values[i]));
    }
    return norm;
}

float safeRatio(float numerator, float denominator)
{
    if (denominator > 0.0f) {
        return numerator / denominator;
    }
    return numerator == 0.0f
        ? 0.0f : std::numeric_limits<float>::infinity();
}

ResidualVerification verifySolution(
    int n,
    const std::unordered_map<std::uint64_t, SymmetricEntryAccumulator>& entries,
    bool symmetrize_input,
    const std::vector<float>& rhs,
    const std::vector<float>& solution,
    const std::vector<float>& expected_solution)
{
    float matrix_norm = 0.0f;
    const std::vector<float> product = symmetricMatrixVectorProduct(
        n, entries, symmetrize_input, solution, &matrix_norm);

    float residual_norm = 0.0f;
    float solution_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        residual_norm = std::max(
            residual_norm,
            std::fabs(product[static_cast<std::size_t>(i)] -
                      rhs[static_cast<std::size_t>(i)]));
        solution_error = std::max(
            solution_error,
            std::fabs(solution[static_cast<std::size_t>(i)] -
                      expected_solution[static_cast<std::size_t>(i)]));
    }

    ResidualVerification verification;
    verification.matrix_infinity_norm = matrix_norm;
    verification.rhs_infinity_norm = infinityNorm(rhs);
    verification.solution_infinity_norm = infinityNorm(solution);
    verification.absolute_residual = residual_norm;
    verification.relative_residual = safeRatio(
        residual_norm, verification.rhs_infinity_norm);
    verification.scaled_backward_error = safeRatio(
        residual_norm,
        verification.matrix_infinity_norm * verification.solution_infinity_norm +
            verification.rhs_infinity_norm);
    verification.relative_solution_error = safeRatio(
        solution_error, infinityNorm(expected_solution));
    return verification;
}

} // namespace

int main()
{
    // 读入后，矩阵在内存中统一使用0-based CSC格式。
    std::vector<int> A_rows;
    std::vector<int> A_cols;
    std::vector<float> A_val;
    int A_n, A_nnz;
    readCSCMatrix("data/A_1215.dat", A_n, A_nnz, A_rows, A_cols, A_val);

    // 分别测试METIS和AMD，选择符号填充元更少的排列。
    const OrderingResult ordering = computeBestOrdering(A_n, A_cols, A_rows);

    std::vector<int> ordered_rows;
    std::vector<int> ordered_cols;
    std::vector<float> ordered_values;
    applySymmetricPermutationCSC(A_n, A_cols, A_rows, A_val, ordering.iperm, ordered_cols, ordered_rows, ordered_values);

    writeCSCMatrix("data/A_1215_ordered.dat", A_n, ordered_rows, ordered_cols, ordered_values);
    std::cout << "Loaded " << A_val.size() << " nonzero entries" << '\n';
    std::cout << A_n << "x" << A_n << ", nonzero = " << A_nnz << '\n';
    std::cout << "METIS symbolic fill-in = " << ordering.metis_fill.fill_entries << ", ratio = " << ordering.metis_fill.fill_ratio << '\n';
    std::cout << "AMD symbolic fill-in = " << ordering.amd_fill.fill_entries << ", ratio = " << ordering.amd_fill.fill_ratio << '\n';
    std::cout << "Selected ordering method = " << orderingMethodName(ordering.method) << '\n';
    std::cout << "Selected symbolic factor L nonzeros = "<< ordering.selected_fill.factor_nonzeros << '\n';
    std::cout << "Reordered nonzeros = " << ordered_values.size() << '\n';

    // 符号分析
    const CholmodSymbolicResult symbolic = analyzeBasicSupernodesWithCholmod(A_n, ordered_cols, ordered_rows);
    writeSymbolicVisualizationData(SUPERNODAL_VISUALIZATION_DIR, A_n, symbolic);
    std::cout << "Elimination tree columns = " << symbolic.column_parent.size() << '\n';
    std::cout << "Basic supernodes = " << symbolic.supernode_parent.size() << '\n';
    std::cout << "Visualization data = " << SUPERNODAL_VISUALIZATION_DIR << '\n';

    // 数值分解：节点分类依据实际候选宽度（包含子节点上传的延迟列）。
    // 1 列节点与 2--64 列节点分别批处理；更大的节点使用 64 列 BK panel。
    GpuLdltOptions gpu_options;
    gpu_options.batched_width_limit = 64;
    gpu_options.panel_size = 64;
    gpu_options.max_batch_nodes = 256;
    gpu_options.large_front_streams = 4;

    GpuSupernodalLdltFactor gpu_factor(gpu_options);
    const GpuLdltStatistics gpu_statistics = gpu_factor.factorize(
        A_n, ordered_cols, ordered_rows, ordered_values, symbolic);

    std::cout << "GPU single-column nodes = "
              << gpu_statistics.single_column_nodes << '\n';
    std::cout << "GPU 2--64 column nodes = "
              << gpu_statistics.batched_small_medium_nodes << '\n';
    std::cout << "GPU >64 column nodes = "
              << gpu_statistics.large_panel_nodes << '\n';
    std::cout << "Accepted 1x1 pivots = "
              << gpu_statistics.accepted_one_by_one_pivots << '\n';
    std::cout << "Accepted 2x2 pivot blocks = "
              << gpu_statistics.accepted_two_by_two_pivots << '\n';
    std::cout << "Delayed pivot events = "
              << gpu_statistics.delayed_columns << '\n';
    std::cout << "Unresolved root columns = "
              << gpu_statistics.unresolved_root_columns << '\n';
    std::cout << "Supernode tree waves = "
              << gpu_statistics.tree_waves << '\n';
    std::cout << "Concurrent large-front CUDA streams = "
              << gpu_statistics.large_front_streams << '\n';
    std::cout << "Maximum input asymmetry = "
              << gpu_statistics.maximum_input_asymmetry << '\n';
    std::cout << "Sorted unique CSC fast path = "
              << (gpu_statistics.sorted_csc_fast_path ? "enabled" : "fallback")
              << '\n';
    std::cout << "CUDA asynchronous memory pool = "
              << (gpu_statistics.asynchronous_memory_pool ? "enabled" : "fallback")
              << '\n';
    std::cout << "Input preprocessing and upload time (ms) = "
              << gpu_statistics.input_preprocessing_milliseconds << '\n';
    std::cout << "Front assembly enqueue time (ms) = "
              << gpu_statistics.front_assembly_milliseconds << '\n';
    std::cout << "Contribution release/barrier time (ms) = "
              << gpu_statistics.contribution_release_milliseconds << '\n';
    std::cout << "Single/small/medium factor time (ms) = "
              << gpu_statistics.small_medium_factorization_milliseconds << '\n';
    std::cout << "Large panel factor time (ms) = "
              << gpu_statistics.large_panel_factorization_milliseconds << '\n';
    std::cout << "Factor save/contribution extraction time (ms) = "
              << gpu_statistics.factor_finalization_milliseconds << '\n';
    const float timed_stage_total =
        gpu_statistics.input_preprocessing_milliseconds +
        gpu_statistics.front_assembly_milliseconds +
        gpu_statistics.contribution_release_milliseconds +
        gpu_statistics.small_medium_factorization_milliseconds +
        gpu_statistics.large_panel_factorization_milliseconds +
        gpu_statistics.factor_finalization_milliseconds;
    std::cout << "Timed stage total (ms) = " << timed_stage_total << '\n';
    std::cout << "Unattributed scheduling time (ms) = "
              << std::max(
                     0.0f,
                     gpu_statistics.factorization_milliseconds - timed_stage_total)
              << '\n';
    std::cout << "GPU factorization time (ms) = "
              << gpu_statistics.factorization_milliseconds << '\n';
    std::cout << "GPU LDLT status = " << gpu_factor.diagnostic() << '\n';

    if (!gpu_factor.complete()) {
        return 2;
    }

    // 文件中没有单独的 RHS，因此用已知解 x_true = 1 构造 b=A*x_true。
    // 这里复用数值分解的上下三角合并与对称化规则，避免验证矩阵不一致。
    const std::unordered_map<std::uint64_t, SymmetricEntryAccumulator>
        symmetric_entries = buildSymmetricEntries(
            A_n, ordered_cols, ordered_rows, ordered_values);
    const std::vector<float> expected_solution(
        static_cast<std::size_t>(A_n), 1.0f);
    const std::vector<float> rhs = symmetricMatrixVectorProduct(
        A_n, symmetric_entries, gpu_options.symmetrize_input,
        expected_solution, 0);

    const std::chrono::steady_clock::time_point solve_start =
        std::chrono::steady_clock::now();
    const std::vector<float> solution = gpu_factor.solve(rhs);
    const std::chrono::steady_clock::time_point solve_end =
        std::chrono::steady_clock::now();
    const float solve_milliseconds =
        std::chrono::duration<float, std::milli>(solve_end - solve_start).count();

    const ResidualVerification verification = verifySolution(
        A_n, symmetric_entries, gpu_options.symmetrize_input,
        rhs, solution, expected_solution);

    std::cout << std::scientific << std::setprecision(8);
    std::cout << "Solve verification x_true = all ones" << '\n';
    std::cout << "GPU solve time (ms) = " << solve_milliseconds << '\n';
    std::cout << "Matrix infinity norm = "
              << verification.matrix_infinity_norm << '\n';
    std::cout << "RHS infinity norm = "
              << verification.rhs_infinity_norm << '\n';
    std::cout << "Solution infinity norm = "
              << verification.solution_infinity_norm << '\n';
    std::cout << "Absolute residual ||A*x-b||_inf = "
              << verification.absolute_residual << '\n';
    std::cout << "Relative residual ||A*x-b||_inf/||b||_inf = "
              << verification.relative_residual << '\n';
    std::cout << "Scaled backward error = "
              << verification.scaled_backward_error << '\n';
    std::cout << "Relative solution error = "
              << verification.relative_solution_error << '\n';

    const float verification_tolerance =
        std::sqrt(std::numeric_limits<float>::epsilon());
    const bool verification_passed =
        std::isfinite(verification.scaled_backward_error) &&
        verification.scaled_backward_error <= verification_tolerance;
    std::cout << "FP32 verification tolerance = "
              << verification_tolerance << '\n';
    std::cout << "Residual verification = "
              << (verification_passed ? "PASS" : "WARNING") << '\n';

    return verification_passed ? 0 : 3;
}
