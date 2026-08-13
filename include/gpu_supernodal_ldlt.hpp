#ifndef SUPERNODAL_GPU_INCLUDE_GPU_SUPERNODAL_LDLT_HPP
#define SUPERNODAL_GPU_INCLUDE_GPU_SUPERNODAL_LDLT_HPP

#include "cholmod_symbolic.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

/** 数值阶段使用的超节点宽度分类。 */
enum class GpuSupernodeClass {
    SingleColumn,
    BatchedSmallMedium,
    LargePanel
};

struct GpuLdltOptions {
    GpuLdltOptions();

    // 实际候选列数（含子节点上传的延迟列）不超过该值时进入批处理。
    int batched_width_limit;

    // 大节点的 BK panel 宽度。遇到延迟主元时会提前刷新并重启 panel。
    int panel_size;

    // 同一消去树波次最多同时实例化的 front 数，限制稠密工作区显存峰值。
    int max_batch_nodes;

    // Bunch--Kaufman 阈值；默认 (1 + sqrt(17)) / 8。
    float bunch_kaufman_gamma;

    // |x| <= max(absolute_pivot_tolerance,
    //             relative_pivot_tolerance * 当前候选块尺度) 时视为数值零。
    float absolute_pivot_tolerance;
    float relative_pivot_tolerance;

    // 2x2 块要求 |det(D)| > two_by_two_tolerance * ||D||_inf^2。
    float two_by_two_tolerance;

    // 输入同时包含上下三角时，先用 (A + A^T) / 2 保证数值对称。
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

    // 输入每列严格有序且无重复时，数值对称化跳过 unordered_map。
    bool sorted_csc_fast_path;

    // CUDA 11.2+ 且设备支持时，临时/因子缓冲区使用默认异步内存池。
    bool asynchronous_memory_pool;

    // 以下分项是宿主端经过时间；异步 front kernel 的完成时间通常记在
    // contribution_release_milliseconds 对应的批次屏障中。
    float input_preprocessing_milliseconds;
    float front_assembly_milliseconds;
    float contribution_release_milliseconds;
    float small_medium_factorization_milliseconds;
    float large_panel_factorization_milliseconds;
    float factor_finalization_milliseconds;

    // 从 CUDA 起止 event 得到的整个数值分解阶段时间。
    float factorization_milliseconds;
    float maximum_input_asymmetry;
};

/**
 * GPU 常驻的 P^T A P = L D L^T 超节点因子。
 *
 * factorize() 使用已经完成的 CHOLMOD 符号分析，不会再次排序或修改超节点。
 * 子节点的延迟列会在数值阶段动态加入父节点候选区。对象可移动、不可复制。
 */
class GpuSupernodalLdltFactor {
public:
    explicit GpuSupernodalLdltFactor(
        const GpuLdltOptions& options = GpuLdltOptions());
    ~GpuSupernodalLdltFactor();

    GpuSupernodalLdltFactor(GpuSupernodalLdltFactor&& other) noexcept;
    GpuSupernodalLdltFactor& operator=(GpuSupernodalLdltFactor&& other) noexcept;

    GpuSupernodalLdltFactor(const GpuSupernodalLdltFactor&) = delete;
    GpuSupernodalLdltFactor& operator=(const GpuSupernodalLdltFactor&) = delete;

    /**
     * 对已重排的 0-based CSC 对称矩阵执行单 GPU 多波前 LDL^T 分解。
     * 数值在本函数开始时一次性上传，之后 contribution 不回传主机。
     */
    GpuLdltStatistics factorize(
        int n,
        const std::vector<int>& col_ptr,
        const std::vector<int>& row_indices,
        const std::vector<float>& values,
        const CholmodSymbolicResult& symbolic);

    /**
     * 使用 GPU 上保留的因子求解重排后系统。仅当 complete() 为 true 时可用。
     * 返回值仍采用重排后的变量编号。
     */
    std::vector<float> solve(const std::vector<float>& reordered_rhs) const;

    bool complete() const;
    const GpuLdltStatistics& statistics() const;
    const std::string& diagnostic() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
