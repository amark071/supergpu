#ifndef SUPERNODAL_GPU_INCLUDE_SYMBOLIC_PIPELINE_HPP
#define SUPERNODAL_GPU_INCLUDE_SYMBOLIC_PIPELINE_HPP

#include "cholmod_symbolic.hpp"
#include "ordering.hpp"

#include <cstddef>
#include <string>
#include <vector>

struct SymbolicPipelineOptions {
    SymbolicPipelineOptions();

    bool compare_orderings;
    bool rebuild_cache;
    bool disable_cache;
    std::string cache_path;
};

struct SymbolicPipelineResult {
    OrderingResult ordering;
    CholmodSymbolicResult symbolic;
    std::vector<int> ordered_col_ptr;
    std::vector<int> ordered_row_indices;
    std::vector<float> ordered_values;

    std::string selected_ordering_method;
    std::size_t selected_factor_nonzeros;
    std::size_t selected_fill_entries;
    double selected_fill_ratio;
    bool cache_hit;

    float ordering_comparison_milliseconds;
    float cholmod_analysis_milliseconds;
    float permutation_milliseconds;
    float total_milliseconds;
};

SymbolicPipelineResult runSymbolicPipeline(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values,
    const SymbolicPipelineOptions& options);

#endif
