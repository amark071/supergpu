#ifndef SUPERNODAL_GPU_INCLUDE_GPU_SUPERNODAL_LDLT_HPP
#define SUPERNODAL_GPU_INCLUDE_GPU_SUPERNODAL_LDLT_HPP

#include "cholmod_symbolic.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

enum class GpuSupernodeClass {
    SingleColumn,
    BatchedSmallMedium,
    LargePanel
};

struct GpuLdltOptions {
    GpuLdltOptions();

    // Candidate columns at or below this width use the batched kernel.
    int batched_width_limit;

    // Maximum number of accepted pivot columns before a large-panel update.
    int panel_size;

    // Maximum number of fronts materialized in one tree-wave batch.
    int max_batch_nodes;

    // Independent large fronts processed concurrently in separate CUDA streams.
    int large_front_streams;

    // Bunch-Kaufman threshold; defaults to (1 + sqrt(17)) / 8.
    float bunch_kaufman_gamma;

    // Values at or below max(absolute, relative * front scale) are numerical zero.
    float absolute_pivot_tolerance;
    float relative_pivot_tolerance;

    // A 2x2 pivot requires |det(D)| > tolerance * ||D||_inf^2.
    float two_by_two_tolerance;

    // Average matching lower/upper entries when both triangles are supplied.
    bool symmetrize_input;
};

struct GpuLdltStatistics {
    GpuLdltStatistics();

    std::size_t single_column_nodes;
    std::size_t batched_small_medium_nodes;
    std::size_t large_panel_nodes;
    std::size_t accepted_one_by_one_pivots;
    std::size_t accepted_two_by_two_pivots;
    std::size_t delayed_columns;
    std::size_t unresolved_root_columns;
    std::size_t tree_waves;
    std::size_t large_front_streams;

    bool sorted_csc_fast_path;
    bool asynchronous_memory_pool;

    float input_preprocessing_milliseconds;
    float front_assembly_milliseconds;
    float contribution_release_milliseconds;
    float small_medium_factorization_milliseconds;
    float large_panel_factorization_milliseconds;
    float factor_finalization_milliseconds;
    float factorization_milliseconds;
    float maximum_input_asymmetry;
};

// GPU-resident P^T A P = L D L^T supernodal factorization.
class GpuSupernodalLdltFactor {
public:
    explicit GpuSupernodalLdltFactor(
        const GpuLdltOptions& options = GpuLdltOptions());
    ~GpuSupernodalLdltFactor();

    GpuSupernodalLdltFactor(GpuSupernodalLdltFactor&& other) noexcept;
    GpuSupernodalLdltFactor& operator=(GpuSupernodalLdltFactor&& other) noexcept;

    GpuSupernodalLdltFactor(const GpuSupernodalLdltFactor&) = delete;
    GpuSupernodalLdltFactor& operator=(const GpuSupernodalLdltFactor&) = delete;

    // The CSC matrix and symbolic structure use reordered, zero-based indices.
    GpuLdltStatistics factorize(
        int n,
        const std::vector<int>& col_ptr,
        const std::vector<int>& row_indices,
        const std::vector<float>& values,
        const CholmodSymbolicResult& symbolic);

    // Solve in reordered coordinates using factors retained on the GPU.
    std::vector<float> solve(const std::vector<float>& reordered_rhs) const;

    bool complete() const;
    const GpuLdltStatistics& statistics() const;
    const std::string& diagnostic() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
