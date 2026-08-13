#include "supernodal_app.hpp"

#include "gpu_supernodal_ldlt.hpp"
#include "io_func.hpp"
#include "solution_verification.hpp"
#include "symbolic_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef SUPERNODAL_VISUALIZATION_DIR
#define SUPERNODAL_VISUALIZATION_DIR "."
#endif

namespace {

struct ApplicationOptions {
    ApplicationOptions()
        : compare_orderings(false),
          export_ordered_matrix(false),
          export_visualization(false),
          rebuild_symbolic_cache(false),
          no_symbolic_cache(false)
    {
    }

    bool compare_orderings;
    bool export_ordered_matrix;
    bool export_visualization;
    bool rebuild_symbolic_cache;
    bool no_symbolic_cache;
};

struct InputMatrix {
    int n;
    int declared_nonzeros;
    std::vector<int> row_indices;
    std::vector<int> col_ptr;
    std::vector<float> values;
};

struct SolveVerificationResult {
    ResidualVerification residual;
    float solve_milliseconds;
    float tolerance;
    bool passed;
};

bool parseCommandLine(
    int argc,
    char** argv,
    ApplicationOptions& options)
{
    for (int argument = 1; argument < argc; ++argument) {
        const std::string option(argv[argument]);
        if (option == "--compare-orderings") {
            options.compare_orderings = true;
        } else if (option == "--export-ordered-matrix") {
            options.export_ordered_matrix = true;
        } else if (option == "--export-visualization") {
            options.export_visualization = true;
        } else if (option == "--rebuild-symbolic-cache") {
            options.rebuild_symbolic_cache = true;
        } else if (option == "--no-symbolic-cache") {
            options.no_symbolic_cache = true;
        } else {
            std::cerr << "Unknown option: " << option << '\n';
            return false;
        }
    }
    return true;
}

InputMatrix loadInputMatrix()
{
    InputMatrix matrix;
    readCSCMatrix(
        "data/A_1215.dat", matrix.n, matrix.declared_nonzeros,
        matrix.row_indices, matrix.col_ptr, matrix.values);
    return matrix;
}

SymbolicPipelineOptions makeSymbolicOptions(
    const ApplicationOptions& options)
{
    SymbolicPipelineOptions symbolic_options;
    symbolic_options.compare_orderings = options.compare_orderings;
    symbolic_options.rebuild_cache = options.rebuild_symbolic_cache;
    symbolic_options.disable_cache = options.no_symbolic_cache;
    return symbolic_options;
}

void exportRequestedSymbolicData(
    const ApplicationOptions& options,
    const InputMatrix& input,
    const SymbolicPipelineResult& pipeline)
{
    if (options.export_ordered_matrix) {
        writeCSCMatrix(
            "data/A_1215_ordered.dat", input.n,
            pipeline.ordered_row_indices, pipeline.ordered_col_ptr,
            pipeline.ordered_values);
    }
    if (options.export_visualization) {
        writeSymbolicVisualizationData(
            SUPERNODAL_VISUALIZATION_DIR, input.n, pipeline.symbolic);
    }
}

const char* symbolicCacheStatus(
    const ApplicationOptions& options,
    const SymbolicPipelineResult& pipeline)
{
    if (options.compare_orderings) {
        return "disabled in comparison mode";
    }
    if (options.no_symbolic_cache) {
        return "disabled by --no-symbolic-cache";
    }
    return pipeline.cache_hit ? "hit" : "miss; cache updated";
}

void printFillStatistics(
    const ApplicationOptions& options,
    const SymbolicPipelineResult& pipeline)
{
    if (options.compare_orderings) {
        std::cout << "METIS symbolic fill-in = "
                  << pipeline.ordering.metis_fill.fill_entries << ", ratio = "
                  << pipeline.ordering.metis_fill.fill_ratio << '\n';
        std::cout << "AMD symbolic fill-in = "
                  << pipeline.ordering.amd_fill.fill_entries << ", ratio = "
                  << pipeline.ordering.amd_fill.fill_ratio << '\n';
        return;
    }
    std::cout << "METIS symbolic fill-in = "
              << pipeline.selected_fill_entries << ", ratio = "
              << pipeline.selected_fill_ratio << '\n';
    std::cout << "AMD symbolic fill-in = skipped (use --compare-orderings)\n";
}

void printSymbolicReport(
    const ApplicationOptions& options,
    const InputMatrix& input,
    const SymbolicPipelineResult& pipeline)
{
    std::cout << "Loaded " << input.values.size() << " nonzero entries\n";
    std::cout << input.n << 'x' << input.n << ", nonzero = "
              << input.declared_nonzeros << '\n';
    printFillStatistics(options, pipeline);
    std::cout << "Selected ordering method = "
              << pipeline.selected_ordering_method << '\n';
    std::cout << "Selected symbolic factor L nonzeros = "
              << pipeline.selected_factor_nonzeros << '\n';
    std::cout << "Reordered nonzeros = "
              << pipeline.ordered_values.size() << '\n';
    std::cout << "Ordering comparison time (ms) = "
              << pipeline.ordering_comparison_milliseconds << '\n';
    std::cout << "Symbolic cache = "
              << symbolicCacheStatus(options, pipeline) << '\n';
    std::cout << "CHOLMOD ordering and symbolic analysis time (ms) = "
              << pipeline.cholmod_analysis_milliseconds << '\n';
    std::cout << "Symmetric CSC permutation time (ms) = "
              << pipeline.permutation_milliseconds << '\n';
    std::cout << "Symbolic pipeline including permutation time (ms) = "
              << pipeline.total_milliseconds << '\n';
    std::cout << "Elimination tree columns = "
              << pipeline.symbolic.column_parent.size() << '\n';
    std::cout << "Basic supernodes = "
              << pipeline.symbolic.supernode_parent.size() << '\n';
    std::cout << "Visualization data = "
              << (options.export_visualization
                      ? SUPERNODAL_VISUALIZATION_DIR
                      : "disabled (use --export-visualization)")
              << '\n';
}

GpuLdltOptions makeGpuOptions()
{
    GpuLdltOptions options;
    options.batched_width_limit = 64;
    options.panel_size = 64;
    options.max_batch_nodes = 256;
    options.large_front_streams = 4;
    return options;
}

float timedStageTotal(const GpuLdltStatistics& statistics)
{
    return statistics.input_preprocessing_milliseconds +
        statistics.front_assembly_milliseconds +
        statistics.contribution_release_milliseconds +
        statistics.small_medium_factorization_milliseconds +
        statistics.large_panel_factorization_milliseconds +
        statistics.factor_finalization_milliseconds;
}

void printFactorCounts(const GpuLdltStatistics& statistics)
{
    std::cout << "GPU single-column nodes = "
              << statistics.single_column_nodes << '\n';
    std::cout << "GPU 2--64 column nodes = "
              << statistics.batched_small_medium_nodes << '\n';
    std::cout << "GPU >64 column nodes = "
              << statistics.large_panel_nodes << '\n';
    std::cout << "Accepted 1x1 pivots = "
              << statistics.accepted_one_by_one_pivots << '\n';
    std::cout << "Accepted 2x2 pivot blocks = "
              << statistics.accepted_two_by_two_pivots << '\n';
    std::cout << "Delayed pivot events = "
              << statistics.delayed_columns << '\n';
    std::cout << "Unresolved root columns = "
              << statistics.unresolved_root_columns << '\n';
    std::cout << "Supernode tree waves = "
              << statistics.tree_waves << '\n';
}

void printGpuFeatureStatus(const GpuLdltStatistics& statistics)
{
    std::cout << "Concurrent large-front CUDA streams = "
              << statistics.large_front_streams << '\n';
    std::cout << "Maximum input asymmetry = "
              << statistics.maximum_input_asymmetry << '\n';
    std::cout << "Sorted unique CSC fast path = "
              << (statistics.sorted_csc_fast_path ? "enabled" : "fallback")
              << '\n';
    std::cout << "CUDA asynchronous memory pool = "
              << (statistics.asynchronous_memory_pool ? "enabled" : "fallback")
              << '\n';
}

void printFactorTimings(const GpuLdltStatistics& statistics)
{
    std::cout << "Input preprocessing and upload time (ms) = "
              << statistics.input_preprocessing_milliseconds << '\n';
    std::cout << "Front assembly enqueue time (ms) = "
              << statistics.front_assembly_milliseconds << '\n';
    std::cout << "Contribution release/barrier time (ms) = "
              << statistics.contribution_release_milliseconds << '\n';
    std::cout << "Single/small/medium factor time (ms) = "
              << statistics.small_medium_factorization_milliseconds << '\n';
    std::cout << "Large panel factor time (ms) = "
              << statistics.large_panel_factorization_milliseconds << '\n';
    std::cout << "Factor save/contribution extraction time (ms) = "
              << statistics.factor_finalization_milliseconds << '\n';
    const float stage_total = timedStageTotal(statistics);
    std::cout << "Timed stage total (ms) = " << stage_total << '\n';
    std::cout << "Unattributed scheduling time (ms) = "
              << std::max(
                     0.0f, statistics.factorization_milliseconds - stage_total)
              << '\n';
    std::cout << "GPU factorization time (ms) = "
              << statistics.factorization_milliseconds << '\n';
}

void printFactorizationReport(
    const GpuLdltStatistics& statistics,
    const GpuSupernodalLdltFactor& factor)
{
    printFactorCounts(statistics);
    printGpuFeatureStatus(statistics);
    printFactorTimings(statistics);
    std::cout << "GPU LDLT status = " << factor.diagnostic() << '\n';
}

SolveVerificationResult runSolveVerification(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values,
    bool symmetrize_input,
    const GpuSupernodalLdltFactor& factor)
{
    const SymmetricEntries symmetric_entries = buildSymmetricEntries(
        n, col_ptr, row_indices, values);
    const std::vector<float> expected_solution(
        static_cast<std::size_t>(n), 1.0f);
    const std::vector<float> rhs = symmetricMatrixVectorProduct(
        n, symmetric_entries, symmetrize_input, expected_solution, 0);

    const std::chrono::steady_clock::time_point solve_begin =
        std::chrono::steady_clock::now();
    const std::vector<float> solution = factor.solve(rhs);

    SolveVerificationResult result;
    result.solve_milliseconds = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - solve_begin).count();
    result.residual = verifySolution(
        n, symmetric_entries, symmetrize_input,
        rhs, solution, expected_solution);
    result.tolerance = std::sqrt(std::numeric_limits<float>::epsilon());
    result.passed = std::isfinite(result.residual.scaled_backward_error) &&
        result.residual.scaled_backward_error <= result.tolerance;
    return result;
}

void printVerificationReport(const SolveVerificationResult& result)
{
    std::cout << std::scientific << std::setprecision(8);
    std::cout << "Solve verification x_true = all ones\n";
    std::cout << "GPU solve time (ms) = " << result.solve_milliseconds << '\n';
    std::cout << "Matrix infinity norm = "
              << result.residual.matrix_infinity_norm << '\n';
    std::cout << "RHS infinity norm = "
              << result.residual.rhs_infinity_norm << '\n';
    std::cout << "Solution infinity norm = "
              << result.residual.solution_infinity_norm << '\n';
    std::cout << "Absolute residual ||A*x-b||_inf = "
              << result.residual.absolute_residual << '\n';
    std::cout << "Relative residual ||A*x-b||_inf/||b||_inf = "
              << result.residual.relative_residual << '\n';
    std::cout << "Scaled backward error = "
              << result.residual.scaled_backward_error << '\n';
    std::cout << "Relative solution error = "
              << result.residual.relative_solution_error << '\n';
    std::cout << "FP32 verification tolerance = " << result.tolerance << '\n';
    std::cout << "Residual verification = "
              << (result.passed ? "PASS" : "WARNING") << '\n';
}

} // namespace

int runSupernodalGpuApplication(int argc, char** argv)
{
    ApplicationOptions options;
    if (!parseCommandLine(argc, argv, options)) {
        return 1;
    }

    const InputMatrix input = loadInputMatrix();
    const SymbolicPipelineResult pipeline = runSymbolicPipeline(
        input.n, input.col_ptr, input.row_indices, input.values,
        makeSymbolicOptions(options));
    exportRequestedSymbolicData(options, input, pipeline);
    printSymbolicReport(options, input, pipeline);

    const GpuLdltOptions gpu_options = makeGpuOptions();
    GpuSupernodalLdltFactor gpu_factor(gpu_options);
    const GpuLdltStatistics gpu_statistics = gpu_factor.factorize(
        input.n, pipeline.ordered_col_ptr, pipeline.ordered_row_indices,
        pipeline.ordered_values, pipeline.symbolic);
    printFactorizationReport(gpu_statistics, gpu_factor);
    if (!gpu_factor.complete()) {
        return 2;
    }

    const SolveVerificationResult verification = runSolveVerification(
        input.n, pipeline.ordered_col_ptr, pipeline.ordered_row_indices,
        pipeline.ordered_values, gpu_options.symmetrize_input, gpu_factor);
    printVerificationReport(verification);
    return verification.passed ? 0 : 3;
}
