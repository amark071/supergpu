#ifndef SUPERNODAL_GPU_INCLUDE_GPU_SUPERNODAL_LU_HPP
#define SUPERNODAL_GPU_INCLUDE_GPU_SUPERNODAL_LU_HPP

#include "unsymmetric_symbolic.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

struct GpuLuOptions {
    GpuLuOptions();

    int batched_width_limit;
    // Dynamic unsymmetric pivots currently require rank-1 panel flushing.
    int panel_size;
    int concurrent_fronts;
    float threshold_pivoting;
    float absolute_pivot_tolerance;
    float relative_pivot_tolerance;
};

struct GpuLuStatistics {
    GpuLuStatistics();

    std::size_t single_column_nodes;
    std::size_t small_medium_nodes;
    std::size_t large_front_nodes;
    std::size_t accepted_pivots;
    std::size_t delayed_columns;
    std::size_t unresolved_root_columns;
    std::size_t tree_waves;
    std::size_t concurrent_fronts;

    float front_assembly_milliseconds;
    float small_medium_factorization_milliseconds;
    float large_front_factorization_milliseconds;
    float factorization_milliseconds;
};

// General sparse PAQ=LU multifrontal factorization. The matrix must already
// use the common reordered coordinates produced by unsymmetric_ordering.
class GpuSupernodalLuFactor {
public:
    explicit GpuSupernodalLuFactor(
        const GpuLuOptions& options = GpuLuOptions());
    ~GpuSupernodalLuFactor();

    GpuSupernodalLuFactor(GpuSupernodalLuFactor&& other) noexcept;
    GpuSupernodalLuFactor& operator=(GpuSupernodalLuFactor&& other) noexcept;
    GpuSupernodalLuFactor(const GpuSupernodalLuFactor&) = delete;
    GpuSupernodalLuFactor& operator=(const GpuSupernodalLuFactor&) = delete;

    GpuLuStatistics factorize(
        int n,
        const std::vector<int>& col_ptr,
        const std::vector<int>& row_indices,
        const std::vector<float>& values,
        const UnsymmetricSymbolicResult& symbolic);

    // rhs is in reordered row coordinates; the result is in reordered column
    // coordinates. The application layer applies the external permutations.
    std::vector<float> solve(const std::vector<float>& reordered_rhs) const;

    bool complete() const;
    const GpuLuStatistics& statistics() const;
    const std::string& diagnostic() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
