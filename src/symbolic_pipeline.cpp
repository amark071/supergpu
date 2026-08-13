#include "symbolic_pipeline.hpp"

#include "symbolic_cache.hpp"

#include <chrono>
#include <cstdint>

namespace {

using Clock = std::chrono::steady_clock;

float elapsedMilliseconds(const Clock::time_point& begin)
{
    return std::chrono::duration<float, std::milli>(
        Clock::now() - begin).count();
}

void collectSelectedFillStatistics(SymbolicPipelineResult& result, int n)
{
    for (std::size_t column = 0;
         column < result.symbolic.column_count.size(); ++column) {
        result.selected_factor_nonzeros += static_cast<std::size_t>(
            result.symbolic.column_count[column]);
    }

    const std::size_t original_triangular_nonzeros =
        static_cast<std::size_t>(n) +
        result.symbolic.input_off_diagonal_nonzeros;
    result.selected_fill_entries = result.selected_factor_nonzeros >=
            original_triangular_nonzeros
        ? result.selected_factor_nonzeros - original_triangular_nonzeros : 0;
    result.selected_fill_ratio = original_triangular_nonzeros == 0
        ? 1.0 : static_cast<double>(result.selected_factor_nonzeros) /
            static_cast<double>(original_triangular_nonzeros);
}

void runOrderingComparison(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    SymbolicPipelineResult& result)
{
    const Clock::time_point begin = Clock::now();
    result.ordering = computeBestOrdering(n, col_ptr, row_indices);
    result.ordering_comparison_milliseconds = elapsedMilliseconds(begin);
    result.selected_ordering_method = orderingMethodName(result.ordering.method);
    result.selected_factor_nonzeros =
        result.ordering.selected_fill.factor_nonzeros;
    result.selected_fill_entries = result.ordering.selected_fill.fill_entries;
    result.selected_fill_ratio = result.ordering.selected_fill.fill_ratio;
}

void runFastSymbolicAnalysis(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const SymbolicPipelineOptions& options,
    SymbolicPipelineResult& result)
{
    const std::uint64_t structure_hash = symbolicStructureHash(
        n, col_ptr, row_indices);
    result.cache_hit = !options.rebuild_cache && !options.disable_cache &&
        loadSymbolicCache(
            options.cache_path, n, row_indices.size(), structure_hash,
            result.symbolic);
    if (!result.cache_hit) {
        const Clock::time_point begin = Clock::now();
        result.symbolic = analyzeAndOrderBasicSupernodesWithCholmod(
            n, col_ptr, row_indices);
        result.cholmod_analysis_milliseconds = elapsedMilliseconds(begin);
        if (!options.disable_cache) {
            saveSymbolicCache(
                options.cache_path, n, row_indices.size(), structure_hash,
                result.symbolic);
        }
    }

    result.ordering.perm = result.symbolic.perm;
    result.ordering.iperm = result.symbolic.iperm;
    result.ordering.method = OrderingMethod::Metis;
    result.selected_ordering_method = "CHOLMOD-METIS fast path";
    collectSelectedFillStatistics(result, n);
}

void applySelectedPermutation(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values,
    SymbolicPipelineResult& result)
{
    const Clock::time_point begin = Clock::now();
    applySymmetricPermutationCSC(
        n, col_ptr, row_indices, values, result.ordering.iperm,
        result.ordered_col_ptr, result.ordered_row_indices,
        result.ordered_values);
    result.permutation_milliseconds = elapsedMilliseconds(begin);
}

void analyzeComparedOrdering(int n, SymbolicPipelineResult& result)
{
    const Clock::time_point begin = Clock::now();
    result.symbolic = analyzeBasicSupernodesWithCholmod(
        n, result.ordered_col_ptr, result.ordered_row_indices);
    result.cholmod_analysis_milliseconds = elapsedMilliseconds(begin);
}

} // namespace

SymbolicPipelineOptions::SymbolicPipelineOptions()
    : compare_orderings(false),
      rebuild_cache(false),
      disable_cache(false)
{
}

SymbolicPipelineResult runSymbolicPipeline(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values,
    const SymbolicPipelineOptions& options)
{
    SymbolicPipelineResult result = {};
    result.selected_fill_ratio = 1.0;
    const Clock::time_point pipeline_begin = Clock::now();

    if (options.compare_orderings) {
        runOrderingComparison(n, col_ptr, row_indices, result);
    } else {
        runFastSymbolicAnalysis(
            n, col_ptr, row_indices, options, result);
    }
    applySelectedPermutation(n, col_ptr, row_indices, values, result);
    if (options.compare_orderings) {
        analyzeComparedOrdering(n, result);
    }
    result.total_milliseconds = elapsedMilliseconds(pipeline_begin);
    return result;
}
