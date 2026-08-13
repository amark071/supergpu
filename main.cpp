#include "cholmod_symbolic.hpp"
#include "gpu_supernodal_ldlt.hpp"
#include "io_func.hpp"
#include "ordering.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
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

const std::uint32_t kSymbolicCacheVersion = 1;

std::uint64_t symbolicStructureHash(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices)
{
    std::uint64_t hash = 1469598103934665603ull;
    const std::uint64_t prime = 1099511628211ull;
    hash = (hash ^ static_cast<std::uint32_t>(n)) * prime;
    for (std::size_t i = 0; i < col_ptr.size(); ++i) {
        hash = (hash ^ static_cast<std::uint32_t>(col_ptr[i])) * prime;
    }
    for (std::size_t i = 0; i < row_indices.size(); ++i) {
        hash = (hash ^ static_cast<std::uint32_t>(row_indices[i])) * prime;
    }
    return hash;
}

template <typename T>
bool readBinary(std::ifstream& input, T& value)
{
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(input);
}

template <typename T>
void writeBinary(std::ofstream& output, const T& value)
{
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

bool readIntVector(
    std::ifstream& input,
    std::vector<int>& values,
    std::uint64_t maximum_size)
{
    std::uint64_t size = 0;
    if (!readBinary(input, size) || size > maximum_size) {
        return false;
    }
    values.resize(static_cast<std::size_t>(size));
    if (size != 0) {
        input.read(
            reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(size * sizeof(int)));
    }
    return static_cast<bool>(input);
}

void writeIntVector(std::ofstream& output, const std::vector<int>& values)
{
    const std::uint64_t size = static_cast<std::uint64_t>(values.size());
    writeBinary(output, size);
    if (!values.empty()) {
        output.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(int)));
    }
}

bool loadSymbolicCache(
    const std::string& path,
    int n,
    std::size_t input_nonzeros,
    std::uint64_t structure_hash,
    CholmodSymbolicResult& symbolic)
{
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input) {
        return false;
    }

    char magic[8] = {};
    input.read(magic, sizeof(magic));
    std::uint32_t version = 0;
    std::int32_t cached_n = 0;
    std::uint64_t cached_nonzeros = 0;
    std::uint64_t cached_hash = 0;
    std::uint64_t input_off_diagonal_nonzeros = 0;
    if (!input || std::memcmp(magic, "SGLDLTSY", 8) != 0 ||
        !readBinary(input, version) ||
        !readBinary(input, cached_n) ||
        !readBinary(input, cached_nonzeros) ||
        !readBinary(input, cached_hash) ||
        !readBinary(input, input_off_diagonal_nonzeros) ||
        version != kSymbolicCacheVersion || cached_n != n ||
        cached_nonzeros != input_nonzeros || cached_hash != structure_hash) {
        return false;
    }

    const std::uint64_t n_size = static_cast<std::uint64_t>(n);
    const std::uint64_t maximum_factor_entries =
        n_size * (n_size + 1) / 2;
    if (!readIntVector(input, symbolic.perm, n_size) ||
        !readIntVector(input, symbolic.iperm, n_size) ||
        !readIntVector(input, symbolic.column_parent, n_size) ||
        !readIntVector(input, symbolic.column_count, n_size) ||
        !readIntVector(input, symbolic.supernode_ptr, n_size + 1) ||
        !readIntVector(input, symbolic.supernode_parent, n_size) ||
        !readIntVector(input, symbolic.row_ptr, n_size + 1) ||
        !readIntVector(
            input, symbolic.supernode_rows, maximum_factor_entries)) {
        return false;
    }
    symbolic.input_off_diagonal_nonzeros =
        static_cast<std::size_t>(input_off_diagonal_nonzeros);
    return symbolic.perm.size() == n_size &&
        symbolic.iperm.size() == n_size &&
        symbolic.column_parent.size() == n_size &&
        symbolic.column_count.size() == n_size &&
        !symbolic.supernode_ptr.empty() &&
        symbolic.supernode_ptr.front() == 0 &&
        symbolic.supernode_ptr.back() == n &&
        symbolic.supernode_parent.size() + 1 ==
            symbolic.supernode_ptr.size() &&
        symbolic.row_ptr.size() == symbolic.supernode_ptr.size() &&
        !symbolic.row_ptr.empty() && symbolic.row_ptr.front() == 0 &&
        symbolic.row_ptr.back() ==
            static_cast<int>(symbolic.supernode_rows.size());
}

void saveSymbolicCache(
    const std::string& path,
    int n,
    std::size_t input_nonzeros,
    std::uint64_t structure_hash,
    const CholmodSymbolicResult& symbolic)
{
    std::ofstream output(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!output) {
        std::cerr << "Warning: could not create symbolic cache: "
                  << path << '\n';
        return;
    }
    output.write("SGLDLTSY", 8);
    writeBinary(output, kSymbolicCacheVersion);
    writeBinary(output, static_cast<std::int32_t>(n));
    writeBinary(output, static_cast<std::uint64_t>(input_nonzeros));
    writeBinary(output, structure_hash);
    writeBinary(
        output,
        static_cast<std::uint64_t>(symbolic.input_off_diagonal_nonzeros));
    writeIntVector(output, symbolic.perm);
    writeIntVector(output, symbolic.iperm);
    writeIntVector(output, symbolic.column_parent);
    writeIntVector(output, symbolic.column_count);
    writeIntVector(output, symbolic.supernode_ptr);
    writeIntVector(output, symbolic.supernode_parent);
    writeIntVector(output, symbolic.row_ptr);
    writeIntVector(output, symbolic.supernode_rows);
}

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

int main(int argc, char** argv)
{
    bool compare_orderings = false;
    bool export_ordered_matrix = false;
    bool export_visualization = false;
    bool rebuild_symbolic_cache = false;
    bool no_symbolic_cache = false;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string option(argv[argument]);
        if (option == "--compare-orderings") {
            compare_orderings = true;
        } else if (option == "--export-ordered-matrix") {
            export_ordered_matrix = true;
        } else if (option == "--export-visualization") {
            export_visualization = true;
        } else if (option == "--rebuild-symbolic-cache") {
            rebuild_symbolic_cache = true;
        } else if (option == "--no-symbolic-cache") {
            no_symbolic_cache = true;
        } else {
            std::cerr << "Unknown option: " << option << '\n';
            return 1;
        }
    }
    // 读入后，矩阵在内存中统一使用0-based CSC格式。
    std::vector<int> A_rows;
    std::vector<int> A_cols;
    std::vector<float> A_val;
    int A_n, A_nnz;
    readCSCMatrix("data/A_1215.dat", A_n, A_nnz, A_rows, A_cols, A_val);

    // 分别测试METIS和AMD，选择符号填充元更少的排列。
    const std::chrono::steady_clock::time_point symbolic_start =
        std::chrono::steady_clock::now();
    OrderingResult ordering;
    CholmodSymbolicResult symbolic;
    float ordering_comparison_milliseconds = 0.0f;
    float cholmod_analysis_milliseconds = 0.0f;
    float permutation_milliseconds = 0.0f;
    std::string selected_ordering_method;
    std::size_t selected_factor_nonzeros = 0;
    std::size_t selected_fill_entries = 0;
    double selected_fill_ratio = 1.0;
    bool symbolic_cache_hit = false;

    if (compare_orderings) {
        const std::chrono::steady_clock::time_point ordering_start =
            std::chrono::steady_clock::now();
        ordering = computeBestOrdering(A_n, A_cols, A_rows);
        ordering_comparison_milliseconds =
            std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - ordering_start).count();
        selected_ordering_method = orderingMethodName(ordering.method);
        selected_factor_nonzeros = ordering.selected_fill.factor_nonzeros;
        selected_fill_entries = ordering.selected_fill.fill_entries;
        selected_fill_ratio = ordering.selected_fill.fill_ratio;
    } else {
        const std::string cache_path = "symbolic_analysis.cache";
        const std::uint64_t structure_hash = symbolicStructureHash(
            A_n, A_cols, A_rows);
        symbolic_cache_hit = !rebuild_symbolic_cache && !no_symbolic_cache &&
            loadSymbolicCache(
            cache_path, A_n, A_rows.size(), structure_hash, symbolic);
        if (!symbolic_cache_hit) {
            const std::chrono::steady_clock::time_point analysis_start =
                std::chrono::steady_clock::now();
            symbolic = analyzeAndOrderBasicSupernodesWithCholmod(
                A_n, A_cols, A_rows);
            cholmod_analysis_milliseconds =
                std::chrono::duration<float, std::milli>(
                    std::chrono::steady_clock::now() - analysis_start).count();
            if (!no_symbolic_cache) {
                saveSymbolicCache(
                    cache_path, A_n, A_rows.size(), structure_hash, symbolic);
            }
        }
        ordering.perm = symbolic.perm;
        ordering.iperm = symbolic.iperm;
        ordering.method = OrderingMethod::Metis;
        selected_ordering_method = "CHOLMOD-METIS fast path";
        for (std::size_t column = 0;
             column < symbolic.column_count.size(); ++column) {
            selected_factor_nonzeros +=
                static_cast<std::size_t>(symbolic.column_count[column]);
        }
        const std::size_t original_triangular_nonzeros =
            static_cast<std::size_t>(A_n) +
            symbolic.input_off_diagonal_nonzeros;
        selected_fill_entries = selected_factor_nonzeros >=
                original_triangular_nonzeros
            ? selected_factor_nonzeros - original_triangular_nonzeros : 0;
        selected_fill_ratio = original_triangular_nonzeros == 0
            ? 1.0 : static_cast<double>(selected_factor_nonzeros) /
                static_cast<double>(original_triangular_nonzeros);
    }

    std::vector<int> ordered_rows;
    std::vector<int> ordered_cols;
    std::vector<float> ordered_values;
    const std::chrono::steady_clock::time_point permutation_start =
        std::chrono::steady_clock::now();
    applySymmetricPermutationCSC(A_n, A_cols, A_rows, A_val, ordering.iperm, ordered_cols, ordered_rows, ordered_values);
    permutation_milliseconds =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - permutation_start).count();

    if (compare_orderings) {
        const std::chrono::steady_clock::time_point analysis_start =
            std::chrono::steady_clock::now();
        symbolic = analyzeBasicSupernodesWithCholmod(
            A_n, ordered_cols, ordered_rows);
        cholmod_analysis_milliseconds =
            std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - analysis_start).count();
    }
    const float symbolic_pipeline_milliseconds =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - symbolic_start).count();

    if (export_ordered_matrix) {
        writeCSCMatrix(
            "data/A_1215_ordered.dat", A_n,
            ordered_rows, ordered_cols, ordered_values);
    }
    std::cout << "Loaded " << A_val.size() << " nonzero entries" << '\n';
    std::cout << A_n << "x" << A_n << ", nonzero = " << A_nnz << '\n';
    if (compare_orderings) {
        std::cout << "METIS symbolic fill-in = "
                  << ordering.metis_fill.fill_entries << ", ratio = "
                  << ordering.metis_fill.fill_ratio << '\n';
        std::cout << "AMD symbolic fill-in = "
                  << ordering.amd_fill.fill_entries << ", ratio = "
                  << ordering.amd_fill.fill_ratio << '\n';
    } else {
        std::cout << "METIS symbolic fill-in = " << selected_fill_entries
                  << ", ratio = " << selected_fill_ratio << '\n';
        std::cout << "AMD symbolic fill-in = skipped (use --compare-orderings)\n";
    }
    std::cout << "Selected ordering method = "
              << selected_ordering_method << '\n';
    std::cout << "Selected symbolic factor L nonzeros = "
              << selected_factor_nonzeros << '\n';
    std::cout << "Reordered nonzeros = " << ordered_values.size() << '\n';
    std::cout << "Ordering comparison time (ms) = "
              << ordering_comparison_milliseconds << '\n';
    std::cout << "Symbolic cache = "
              << (compare_orderings
                      ? "disabled in comparison mode"
                      : (no_symbolic_cache
                          ? "disabled by --no-symbolic-cache"
                          : (symbolic_cache_hit
                              ? "hit" : "miss; cache updated")))
              << '\n';
    std::cout << "CHOLMOD ordering and symbolic analysis time (ms) = "
              << cholmod_analysis_milliseconds << '\n';
    std::cout << "Symmetric CSC permutation time (ms) = "
              << permutation_milliseconds << '\n';
    std::cout << "Symbolic pipeline including permutation time (ms) = "
              << symbolic_pipeline_milliseconds << '\n';

    // 符号分析
    if (export_visualization) {
        writeSymbolicVisualizationData(
            SUPERNODAL_VISUALIZATION_DIR, A_n, symbolic);
    }
    std::cout << "Elimination tree columns = " << symbolic.column_parent.size() << '\n';
    std::cout << "Basic supernodes = " << symbolic.supernode_parent.size() << '\n';
    std::cout << "Visualization data = "
              << (export_visualization
                      ? SUPERNODAL_VISUALIZATION_DIR
                      : "disabled (use --export-visualization)")
              << '\n';

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
