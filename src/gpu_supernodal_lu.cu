#include "gpu_supernodal_lu.hpp"

#include "gpu_supernodal_lu_kernels.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <future>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using LuScalar = unsymmetric_lu_kernels::Scalar;

void checkCuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void checkCublas(cublasStatus_t status, const char* operation)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with cuBLAS status " +
            std::to_string(static_cast<int>(status)));
    }
}

template <class T>
class DeviceArray {
public:
    DeviceArray() : pointer_(0), size_(0) {}
    explicit DeviceArray(std::size_t size) : pointer_(0), size_(size)
    {
        if (size_ != 0) {
            checkCuda(cudaMalloc(
                reinterpret_cast<void**>(&pointer_), size_ * sizeof(T)),
                "cudaMalloc");
        }
    }
    ~DeviceArray()
    {
        if (pointer_ != 0) {
            cudaFree(pointer_);
        }
    }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    T* get() { return pointer_; }
    const T* get() const { return pointer_; }
    std::size_t size() const { return size_; }

private:
    T* pointer_;
    std::size_t size_;
};

struct FrontData {
    int node = -1;
    int candidate_count = 0;
    std::vector<int> row_ids;
    std::vector<int> col_ids;
    std::vector<LuScalar> matrix;
};

struct FactorData {
    int candidate_count = 0;
    int accepted = 0;
    int delayed = 0;
    std::vector<int> row_ids;
    std::vector<int> col_ids;
    // Full frontal matrix is retained only until its Schur contribution has
    // been assembled into the parent. The solve data then lives in the two
    // compact arrays below.
    std::vector<LuScalar> matrix;
    std::vector<LuScalar> factor_columns;
    std::vector<LuScalar> upper_trailing;
    bool compacted = false;
};

struct GpuFrontResult {
    int node = -1;
    FactorData factor;
    bool large = false;
    double milliseconds = 0.0;
};

void validateCsc(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<double>& values)
{
    if (n < 0 || col_ptr.size() != static_cast<std::size_t>(n + 1) ||
        col_ptr.empty() || col_ptr.front() != 0 ||
        col_ptr.back() != static_cast<int>(row_indices.size()) ||
        values.size() != row_indices.size()) {
        throw std::invalid_argument("invalid CSC matrix for general GPU LU");
    }
    for (int col = 0; col < n; ++col) {
        if (col_ptr[static_cast<std::size_t>(col)] >
            col_ptr[static_cast<std::size_t>(col + 1)]) {
            throw std::invalid_argument("CSC column pointers are not monotonic");
        }
    }
    for (std::size_t p = 0; p < row_indices.size(); ++p) {
        if (row_indices[p] < 0 || row_indices[p] >= n) {
            throw std::invalid_argument("CSC row index is out of range");
        }
    }
}

void validateSymbolic(int n, const UnsymmetricSymbolicResult& symbolic)
{
    const int nodes = static_cast<int>(symbolic.supernode_parent.size());
    if (symbolic.column_parent.size() != static_cast<std::size_t>(n) ||
        symbolic.supernode_ptr.size() != static_cast<std::size_t>(nodes + 1) ||
        symbolic.front_ptr.size() != static_cast<std::size_t>(nodes + 1) ||
        symbolic.supernode_ptr.empty() || symbolic.supernode_ptr.front() != 0 ||
        symbolic.supernode_ptr.back() != n || symbolic.front_ptr.front() != 0 ||
        symbolic.front_ptr.back() !=
            static_cast<int>(symbolic.front_indices.size())) {
        throw std::invalid_argument("invalid unsymmetric symbolic structure");
    }
    for (int node = 0; node < nodes; ++node) {
        const int parent = symbolic.supernode_parent[static_cast<std::size_t>(node)];
        if (parent >= 0 && (parent <= node || parent >= nodes)) {
            throw std::invalid_argument("supernode tree is not in postorder");
        }
    }
}

LuScalar frontScale(const FrontData& front)
{
    LuScalar scale = 0.0;
    for (std::size_t i = 0; i < front.matrix.size(); ++i) {
        scale = std::max(scale, std::fabs(front.matrix[i]));
    }
    return scale;
}

std::size_t estimatedTrailingUpdates(const FrontData& front)
{
    const int rows = static_cast<int>(front.row_ids.size());
    const int cols = static_cast<int>(front.col_ids.size());
    std::size_t updates = 0;
    for (int step = 0; step < front.candidate_count; ++step) {
        updates += static_cast<std::size_t>(rows - step - 1) *
            static_cast<std::size_t>(cols - step - 1);
    }
    return updates;
}

void copyToDeviceAsync(
    DeviceArray<LuScalar>& device_matrix,
    DeviceArray<int>& device_rows,
    DeviceArray<int>& device_cols,
    const FrontData& front,
    cudaStream_t stream)
{
    checkCuda(cudaMemcpyAsync(
        device_matrix.get(), front.matrix.data(),
        front.matrix.size() * sizeof(LuScalar), cudaMemcpyHostToDevice, stream),
        "copy front matrix to GPU");
    checkCuda(cudaMemcpyAsync(
        device_rows.get(), front.row_ids.data(),
        front.row_ids.size() * sizeof(int), cudaMemcpyHostToDevice, stream),
        "copy front row ids to GPU");
    checkCuda(cudaMemcpyAsync(
        device_cols.get(), front.col_ids.data(),
        front.col_ids.size() * sizeof(int), cudaMemcpyHostToDevice, stream),
        "copy front column ids to GPU");
}

void copyFromDeviceAsync(
    FactorData& factor,
    const DeviceArray<LuScalar>& device_matrix,
    const DeviceArray<int>& device_rows,
    const DeviceArray<int>& device_cols,
    cudaStream_t stream)
{
    checkCuda(cudaMemcpyAsync(
        factor.matrix.data(), device_matrix.get(),
        factor.matrix.size() * sizeof(LuScalar), cudaMemcpyDeviceToHost, stream),
        "copy LU front from GPU");
    checkCuda(cudaMemcpyAsync(
        factor.row_ids.data(), device_rows.get(),
        factor.row_ids.size() * sizeof(int), cudaMemcpyDeviceToHost, stream),
        "copy LU row ids from GPU");
    checkCuda(cudaMemcpyAsync(
        factor.col_ids.data(), device_cols.get(),
        factor.col_ids.size() * sizeof(int), cudaMemcpyDeviceToHost, stream),
        "copy LU column ids from GPU");
}

void factorSmallFrontBatch(
    std::vector<FrontData>& fronts,
    const std::vector<int>& front_indices,
    const GpuLuOptions& options,
    std::vector<GpuFrontResult>& results)
{
    if (front_indices.empty()) {
        return;
    }
    typedef unsymmetric_lu_kernels::SmallFrontDescriptor Descriptor;
    std::vector<std::size_t> matrix_offsets(front_indices.size(), 0);
    std::vector<std::size_t> row_offsets(front_indices.size(), 0);
    std::vector<std::size_t> col_offsets(front_indices.size(), 0);
    std::vector<std::size_t> matrix_sizes(front_indices.size(), 0);
    std::vector<std::size_t> row_sizes(front_indices.size(), 0);
    std::vector<std::size_t> col_sizes(front_indices.size(), 0);
    std::vector<int> candidate_counts(front_indices.size(), 0);
    std::vector<LuScalar> scales(front_indices.size(), 0.0);
    std::size_t matrix_count = 0;
    std::size_t row_count = 0;
    std::size_t col_count = 0;
    for (std::size_t i = 0; i < front_indices.size(); ++i) {
        const FrontData& front = fronts[static_cast<std::size_t>(front_indices[i])];
        matrix_offsets[i] = matrix_count;
        row_offsets[i] = row_count;
        col_offsets[i] = col_count;
        matrix_sizes[i] = front.matrix.size();
        row_sizes[i] = front.row_ids.size();
        col_sizes[i] = front.col_ids.size();
        candidate_counts[i] = front.candidate_count;
        scales[i] = frontScale(front);
        matrix_count += matrix_sizes[i];
        row_count += row_sizes[i];
        col_count += col_sizes[i];
    }

    std::vector<LuScalar> host_matrix(matrix_count);
    std::vector<int> host_rows(row_count);
    std::vector<int> host_cols(col_count);
    for (std::size_t i = 0; i < front_indices.size(); ++i) {
        FrontData& front = fronts[static_cast<std::size_t>(front_indices[i])];
        std::copy(front.matrix.begin(), front.matrix.end(),
                  host_matrix.begin() + matrix_offsets[i]);
        std::copy(front.row_ids.begin(), front.row_ids.end(),
                  host_rows.begin() + row_offsets[i]);
        std::copy(front.col_ids.begin(), front.col_ids.end(),
                  host_cols.begin() + col_offsets[i]);
        std::vector<LuScalar>().swap(front.matrix);
        std::vector<int>().swap(front.row_ids);
        std::vector<int>().swap(front.col_ids);
    }

    DeviceArray<LuScalar> device_matrix(matrix_count);
    DeviceArray<int> device_rows(row_count);
    DeviceArray<int> device_cols(col_count);
    DeviceArray<Descriptor> device_descriptors(front_indices.size());
    std::vector<Descriptor> descriptors(front_indices.size());
    for (std::size_t i = 0; i < front_indices.size(); ++i) {
        Descriptor descriptor;
        descriptor.matrix = device_matrix.get() + matrix_offsets[i];
        descriptor.row_ids = device_rows.get() + row_offsets[i];
        descriptor.col_ids = device_cols.get() + col_offsets[i];
        descriptor.row_count = static_cast<int>(row_sizes[i]);
        descriptor.col_count = static_cast<int>(col_sizes[i]);
        descriptor.candidate_count = candidate_counts[i];
        descriptor.threshold = options.threshold_pivoting;
        descriptor.zero_tolerance = std::max(
            static_cast<LuScalar>(options.absolute_pivot_tolerance),
            static_cast<LuScalar>(options.relative_pivot_tolerance) *
                scales[i]);
        descriptor.accepted = 0;
        descriptor.delayed = 0;
        descriptors[i] = descriptor;
    }

    checkCuda(cudaMemcpy(
        device_matrix.get(), host_matrix.data(),
        matrix_count * sizeof(LuScalar), cudaMemcpyHostToDevice),
        "upload batched LU matrices");
    checkCuda(cudaMemcpy(
        device_rows.get(), host_rows.data(),
        row_count * sizeof(int), cudaMemcpyHostToDevice),
        "upload batched LU row ids");
    checkCuda(cudaMemcpy(
        device_cols.get(), host_cols.data(),
        col_count * sizeof(int), cudaMemcpyHostToDevice),
        "upload batched LU column ids");
    checkCuda(cudaMemcpy(
        device_descriptors.get(), descriptors.data(),
        descriptors.size() * sizeof(Descriptor), cudaMemcpyHostToDevice),
        "upload batched LU descriptors");

    unsymmetric_lu_kernels::factorSmallFronts<<<
        static_cast<unsigned int>(descriptors.size()), 256>>>(
            device_descriptors.get());
    checkCuda(cudaGetLastError(), "launch batched small/medium LU");
    checkCuda(cudaMemcpy(
        descriptors.data(), device_descriptors.get(),
        descriptors.size() * sizeof(Descriptor), cudaMemcpyDeviceToHost),
        "download batched LU descriptors");
    checkCuda(cudaMemcpy(
        host_matrix.data(), device_matrix.get(),
        matrix_count * sizeof(LuScalar), cudaMemcpyDeviceToHost),
        "download batched LU matrices");
    checkCuda(cudaMemcpy(
        host_rows.data(), device_rows.get(),
        row_count * sizeof(int), cudaMemcpyDeviceToHost),
        "download batched LU row ids");
    checkCuda(cudaMemcpy(
        host_cols.data(), device_cols.get(),
        col_count * sizeof(int), cudaMemcpyDeviceToHost),
        "download batched LU column ids");

    for (std::size_t i = 0; i < front_indices.size(); ++i) {
        const int index = front_indices[i];
        GpuFrontResult& result = results[static_cast<std::size_t>(index)];
        result.node = fronts[static_cast<std::size_t>(index)].node;
        result.large = false;
        result.factor.candidate_count = candidate_counts[i];
        result.factor.accepted = descriptors[i].accepted;
        result.factor.delayed = descriptors[i].delayed;
        result.factor.matrix.assign(
            host_matrix.begin() + matrix_offsets[i],
            host_matrix.begin() + matrix_offsets[i] + matrix_sizes[i]);
        result.factor.row_ids.assign(
            host_rows.begin() + row_offsets[i],
            host_rows.begin() + row_offsets[i] + row_sizes[i]);
        result.factor.col_ids.assign(
            host_cols.begin() + col_offsets[i],
            host_cols.begin() + col_offsets[i] + col_sizes[i]);
    }
}

void flushLargePanel(
    DeviceArray<LuScalar>& matrix,
    int row_count,
    int col_count,
    int panel_begin,
    int accepted_end,
    int trailing_col_begin,
    cublasHandle_t cublas)
{
    const int width = accepted_end - panel_begin;
    const int right_cols = col_count - trailing_col_begin;
    if (width <= 0 || right_cols <= 0) {
        return;
    }
    const LuScalar one = 1.0;
    checkCublas(cublasDtrsm(
        cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N, CUBLAS_DIAG_UNIT, width, right_cols, &one,
        matrix.get() + panel_begin + panel_begin * row_count, row_count,
        matrix.get() + panel_begin + trailing_col_begin * row_count, row_count),
        "cuBLAS LU panel TRSM");

    const int trailing_rows = row_count - accepted_end;
    if (trailing_rows <= 0) {
        return;
    }
    const LuScalar minus_one = -1.0;
    checkCublas(cublasDgemm(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        trailing_rows, right_cols, width, &minus_one,
        matrix.get() + accepted_end + panel_begin * row_count, row_count,
        matrix.get() + panel_begin + trailing_col_begin * row_count, row_count,
        &one,
        matrix.get() + accepted_end + trailing_col_begin * row_count,
        row_count),
        "cuBLAS LU trailing GEMM");
}

void runLargeFront(
    DeviceArray<LuScalar>& matrix,
    DeviceArray<int>& rows,
    DeviceArray<int>& cols,
    int row_count,
    int col_count,
    int candidate_count,
    const GpuLuOptions& options,
    LuScalar zero_tolerance,
    int& accepted,
    int& delayed,
    cudaStream_t stream)
{
    cublasHandle_t cublas = 0;
    checkCublas(cublasCreate(&cublas), "create LU cuBLAS handle");
    try {
        checkCublas(cublasSetStream(cublas, stream), "set LU cuBLAS stream");
        DeviceArray<unsymmetric_lu_kernels::LargePanelState> device_state(1);
        int step = 0;
        int active_end = candidate_count;
        while (step < active_end) {
            const int panel_begin = step;
            unsymmetric_lu_kernels::factorLargePanel<<<1, 256, 0, stream>>>(
                matrix.get(), row_count, row_count, col_count,
                candidate_count, rows.get(), cols.get(), panel_begin,
                active_end, options.panel_size, options.threshold_pivoting,
                zero_tolerance, device_state.get());
            checkCuda(cudaGetLastError(), "launch GPU-resident LU panel");
            unsymmetric_lu_kernels::LargePanelState host_state;
            checkCuda(cudaMemcpyAsync(
                &host_state, device_state.get(), sizeof(host_state),
                cudaMemcpyDeviceToHost, stream),
                "download LU panel summary");
            checkCuda(cudaStreamSynchronize(stream),
                "wait for LU panel summary");
            if (host_state.step < panel_begin ||
                host_state.step > candidate_count ||
                host_state.active_end < host_state.step ||
                host_state.active_end > active_end ||
                host_state.updated_col_end < host_state.step ||
                host_state.updated_col_end > col_count) {
                throw std::runtime_error("invalid GPU LU panel state");
            }
            step = host_state.step;
            active_end = host_state.active_end;
            flushLargePanel(
                matrix, row_count, col_count,
                panel_begin, step, host_state.updated_col_end, cublas);
            checkCuda(cudaStreamSynchronize(stream), "flush large LU panel");
            if (step == panel_begin && step >= active_end) {
                break;
            }
        }
        accepted = step;
        delayed = candidate_count - accepted;
    } catch (...) {
        cublasDestroy(cublas);
        throw;
    }
    checkCublas(cublasDestroy(cublas), "destroy LU cuBLAS handle");
}

GpuFrontResult factorLargeFrontOnGpu(
    FrontData front,
    const GpuLuOptions& options)
{
    const std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    const int row_count = static_cast<int>(front.row_ids.size());
    const int col_count = static_cast<int>(front.col_ids.size());
    if (row_count < front.candidate_count || col_count < front.candidate_count) {
        throw std::runtime_error("front has fewer rows or columns than candidates");
    }

    cudaStream_t stream = 0;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "create LU front stream");
    try {
        DeviceArray<LuScalar> device_matrix(front.matrix.size());
        DeviceArray<int> device_rows(front.row_ids.size());
        DeviceArray<int> device_cols(front.col_ids.size());
        copyToDeviceAsync(
            device_matrix, device_rows, device_cols, front, stream);

        GpuFrontResult result;
        result.node = front.node;
        result.large = true;
        result.factor.candidate_count = front.candidate_count;
        const LuScalar zero_tolerance = std::max(
            static_cast<LuScalar>(options.absolute_pivot_tolerance),
            static_cast<LuScalar>(options.relative_pivot_tolerance) *
                frontScale(front));
        runLargeFront(
            device_matrix, device_rows, device_cols, row_count, col_count,
            front.candidate_count, options, zero_tolerance,
            result.factor.accepted, result.factor.delayed, stream);
        result.factor.row_ids.resize(front.row_ids.size());
        result.factor.col_ids.resize(front.col_ids.size());
        result.factor.matrix.resize(front.matrix.size());
        copyFromDeviceAsync(
            result.factor, device_matrix, device_rows, device_cols, stream);
        checkCuda(cudaStreamSynchronize(stream), "download factored LU front");
        result.milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        checkCuda(cudaStreamDestroy(stream), "destroy LU front stream");
        return result;
    } catch (...) {
        cudaStreamDestroy(stream);
        throw;
    }
}

} // namespace

GpuLuOptions::GpuLuOptions()
    : batched_width_limit(64),
      batched_update_limit(4000000),
      max_batch_nodes(256),
      max_batch_bytes(256ULL * 1024ULL * 1024ULL),
      panel_size(32),
      concurrent_fronts(4),
      threshold_pivoting(0.1),
      absolute_pivot_tolerance(0.0),
      relative_pivot_tolerance(
          32.0 * std::numeric_limits<double>::epsilon())
{
}

GpuLuStatistics::GpuLuStatistics()
    : single_column_nodes(0), small_medium_nodes(0), large_front_nodes(0),
      accepted_pivots(0), delayed_columns(0), unresolved_root_columns(0),
      tree_waves(0), concurrent_fronts(0),
      maximum_batch_nodes(0), maximum_batch_bytes(0),
      compact_input_bucket_bytes(0), released_contribution_entries(0),
      front_assembly_milliseconds(0.0),
      small_medium_factorization_milliseconds(0.0),
      large_front_factorization_milliseconds(0.0),
      factorization_milliseconds(0.0)
{
}

class GpuSupernodalLuFactor::Impl {
public:
    explicit Impl(const GpuLuOptions& options)
        : options_(options), n_(0), complete_(false)
    {
        if (options_.batched_width_limit < 1 ||
            options_.batched_update_limit == 0 ||
            options_.max_batch_nodes == 0 || options_.max_batch_bytes == 0 ||
            options_.panel_size < 1 ||
            options_.concurrent_fronts < 1 ||
            options_.threshold_pivoting <= 0.0 ||
            options_.threshold_pivoting > 1.0) {
            throw std::invalid_argument("invalid general GPU LU options");
        }
    }

    GpuLuStatistics factorize(
        int n,
        const std::vector<int>& col_ptr,
        const std::vector<int>& row_indices,
        const std::vector<double>& values,
        const UnsymmetricSymbolicResult& symbolic)
    {
        validateCsc(n, col_ptr, row_indices, values);
        validateSymbolic(n, symbolic);
        reset(n, symbolic);
        checkCuda(cudaFree(0), "initialize CUDA runtime for general LU");
        const std::chrono::steady_clock::time_point factor_begin =
            std::chrono::steady_clock::now();

        buildInputBuckets(col_ptr, row_indices, values, symbolic);
        const std::vector<std::vector<int> > waves = buildTreeWaves();
        statistics_.tree_waves = waves.size();
        statistics_.concurrent_fronts = static_cast<std::size_t>(
            std::min(options_.concurrent_fronts,
                     waves.empty() ? 0 : maximumWaveSize(waves)));
        for (std::size_t wave = 0; wave < waves.size(); ++wave) {
            processWave(waves[wave], symbolic);
        }
        releaseInputBuckets();

        std::size_t accepted_total = 0;
        for (std::size_t node = 0; node < factors_.size(); ++node) {
            accepted_total += static_cast<std::size_t>(factors_[node].accepted);
            if (symbolic.supernode_parent[node] < 0) {
                statistics_.unresolved_root_columns +=
                    static_cast<std::size_t>(factors_[node].delayed);
            }
        }
        complete_ = accepted_total == static_cast<std::size_t>(n_) &&
            statistics_.unresolved_root_columns == 0;
        diagnostic_ = complete_
            ? "complete PAQ=LU factorization"
            : "numerically singular: unresolved delayed columns at a root";
        statistics_.factorization_milliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - factor_begin).count();
        return statistics_;
    }

    std::vector<double> solve(const std::vector<double>& rhs) const
    {
        if (!complete_) {
            throw std::runtime_error("cannot solve with an incomplete general LU");
        }
        if (rhs.size() != static_cast<std::size_t>(n_)) {
            throw std::invalid_argument("general LU right-hand side size mismatch");
        }
        std::vector<LuScalar> work(rhs.begin(), rhs.end());
        forwardSolve(work);
        return backwardSolve(work);
    }

    bool complete() const { return complete_; }
    const GpuLuStatistics& statistics() const { return statistics_; }
    const std::string& diagnostic() const { return diagnostic_; }

private:
    void reset(int n, const UnsymmetricSymbolicResult& symbolic)
    {
        n_ = n;
        complete_ = false;
        diagnostic_.clear();
        statistics_ = GpuLuStatistics();
        factors_.assign(symbolic.supernode_parent.size(), FactorData());
        input_rows_ = 0;
        input_values_ = 0;
        input_owner_ptr_.clear();
        input_entry_indices_.clear();
        input_entry_columns_.clear();
        row_position_.assign(static_cast<std::size_t>(n_), -1);
        col_position_.assign(static_cast<std::size_t>(n_), -1);
        children_.assign(
            symbolic.supernode_parent.size(), std::vector<int>());
        for (std::size_t node = 0; node < symbolic.supernode_parent.size(); ++node) {
            const int parent = symbolic.supernode_parent[node];
            if (parent >= 0) {
                children_[static_cast<std::size_t>(parent)].push_back(
                    static_cast<int>(node));
            }
        }
    }

    void buildInputBuckets(
        const std::vector<int>& col_ptr,
        const std::vector<int>& row_indices,
        const std::vector<double>& values,
        const UnsymmetricSymbolicResult& symbolic)
    {
        input_rows_ = &row_indices;
        input_values_ = &values;
        std::vector<int> column_node(static_cast<std::size_t>(n_), -1);
        for (std::size_t node = 0; node < factors_.size(); ++node) {
            const int begin = symbolic.supernode_ptr[node];
            const int end = symbolic.supernode_ptr[node + 1];
            for (int col = begin; col < end; ++col) {
                column_node[static_cast<std::size_t>(col)] =
                    static_cast<int>(node);
            }
        }
        input_owner_ptr_.assign(factors_.size() + 1, 0);
        for (int col = 0; col < n_; ++col) {
            for (int p = col_ptr[static_cast<std::size_t>(col)];
                 p < col_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
                const int row = row_indices[static_cast<std::size_t>(p)];
                const int owner_col = std::min(row, col);
                const int owner = column_node[static_cast<std::size_t>(owner_col)];
                ++input_owner_ptr_[static_cast<std::size_t>(owner + 1)];
            }
        }
        for (std::size_t node = 0; node < factors_.size(); ++node) {
            input_owner_ptr_[node + 1] += input_owner_ptr_[node];
        }
        input_entry_indices_.resize(row_indices.size());
        input_entry_columns_.resize(row_indices.size());
        std::vector<int> next = input_owner_ptr_;
        for (int col = 0; col < n_; ++col) {
            for (int p = col_ptr[static_cast<std::size_t>(col)];
                 p < col_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
                const int row = row_indices[static_cast<std::size_t>(p)];
                const int owner_col = std::min(row, col);
                const int owner = column_node[static_cast<std::size_t>(owner_col)];
                const int destination = next[static_cast<std::size_t>(owner)]++;
                input_entry_indices_[static_cast<std::size_t>(destination)] = p;
                input_entry_columns_[static_cast<std::size_t>(destination)] = col;
            }
        }
        statistics_.compact_input_bucket_bytes =
            input_owner_ptr_.size() * sizeof(int) +
            input_entry_indices_.size() * sizeof(int) +
            input_entry_columns_.size() * sizeof(int);
    }

    void releaseInputBuckets()
    {
        input_rows_ = 0;
        input_values_ = 0;
        std::vector<int>().swap(input_owner_ptr_);
        std::vector<int>().swap(input_entry_indices_);
        std::vector<int>().swap(input_entry_columns_);
    }

    std::vector<std::vector<int> > buildTreeWaves() const
    {
        std::vector<int> level(factors_.size(), 0);
        int maximum_level = -1;
        for (std::size_t node = 0; node < factors_.size(); ++node) {
            int node_level = 0;
            for (std::size_t child = 0; child < children_[node].size(); ++child) {
                node_level = std::max(
                    node_level,
                    level[static_cast<std::size_t>(children_[node][child])] + 1);
            }
            level[node] = node_level;
            maximum_level = std::max(maximum_level, node_level);
        }
        std::vector<std::vector<int> > waves(
            static_cast<std::size_t>(maximum_level + 1));
        for (std::size_t node = 0; node < level.size(); ++node) {
            waves[static_cast<std::size_t>(level[node])].push_back(
                static_cast<int>(node));
        }
        return waves;
    }

    static int maximumWaveSize(const std::vector<std::vector<int> >& waves)
    {
        std::size_t result = 0;
        for (std::size_t i = 0; i < waves.size(); ++i) {
            result = std::max(result, waves[i].size());
        }
        return static_cast<int>(result);
    }

    static void appendUnique(
        int id, std::vector<int>& ids, std::vector<int>& positions)
    {
        if (positions[static_cast<std::size_t>(id)] < 0) {
            positions[static_cast<std::size_t>(id)] =
                static_cast<int>(ids.size());
            ids.push_back(id);
        }
    }

    static void appendCandidate(
        int id, std::vector<int>& ids, std::vector<int>& positions)
    {
        if (positions[static_cast<std::size_t>(id)] >= 0) {
            throw std::runtime_error("duplicate fully-summed LU candidate");
        }
        positions[static_cast<std::size_t>(id)] = static_cast<int>(ids.size());
        ids.push_back(id);
    }

    void compactFactor(FactorData& factor)
    {
        if (factor.compacted) {
            return;
        }
        const std::size_t rows = factor.row_ids.size();
        const std::size_t cols = factor.col_ids.size();
        const std::size_t accepted = static_cast<std::size_t>(factor.accepted);
        if (accepted > rows || accepted > cols ||
            factor.matrix.size() != rows * cols) {
            throw std::runtime_error("invalid LU front during compaction");
        }

        factor.factor_columns.assign(
            factor.matrix.begin(),
            factor.matrix.begin() + rows * accepted);
        factor.upper_trailing.resize(accepted * (cols - accepted));
        for (std::size_t col = accepted; col < cols; ++col) {
            std::copy(
                factor.matrix.begin() + col * rows,
                factor.matrix.begin() + col * rows + accepted,
                factor.upper_trailing.begin() + (col - accepted) * accepted);
        }
        statistics_.released_contribution_entries +=
            (rows - accepted) * (cols - accepted);
        std::vector<LuScalar>().swap(factor.matrix);
        factor.compacted = true;
    }

    FrontData prepareFront(
        int node,
        const UnsymmetricSymbolicResult& symbolic)
    {
        FrontData front;
        front.node = node;
        const int owned_begin = symbolic.supernode_ptr[static_cast<std::size_t>(node)];
        const int owned_end = symbolic.supernode_ptr[static_cast<std::size_t>(node + 1)];
        for (int id = owned_begin; id < owned_end; ++id) {
            appendCandidate(id, front.row_ids, row_position_);
            appendCandidate(id, front.col_ids, col_position_);
        }
        for (std::size_t c = 0; c < children_[static_cast<std::size_t>(node)].size(); ++c) {
            const FactorData& child = factors_[static_cast<std::size_t>(
                children_[static_cast<std::size_t>(node)][c])];
            for (int k = 0; k < child.delayed; ++k) {
                appendCandidate(
                    child.row_ids[static_cast<std::size_t>(child.accepted + k)],
                    front.row_ids, row_position_);
                appendCandidate(
                    child.col_ids[static_cast<std::size_t>(child.accepted + k)],
                    front.col_ids, col_position_);
            }
        }
        if (front.row_ids.size() != front.col_ids.size()) {
            throw std::runtime_error("delayed row and column candidate counts differ");
        }
        front.candidate_count = static_cast<int>(front.row_ids.size());

        const int front_begin = symbolic.front_ptr[static_cast<std::size_t>(node)];
        const int front_end = symbolic.front_ptr[static_cast<std::size_t>(node + 1)];
        for (int p = front_begin; p < front_end; ++p) {
            const int id = symbolic.front_indices[static_cast<std::size_t>(p)];
            appendUnique(id, front.row_ids, row_position_);
            appendUnique(id, front.col_ids, col_position_);
        }
        for (std::size_t c = 0; c < children_[static_cast<std::size_t>(node)].size(); ++c) {
            const FactorData& child = factors_[static_cast<std::size_t>(
                children_[static_cast<std::size_t>(node)][c])];
            for (std::size_t k = static_cast<std::size_t>(child.accepted);
                 k < child.row_ids.size(); ++k) {
                appendUnique(child.row_ids[k], front.row_ids, row_position_);
            }
            for (std::size_t k = static_cast<std::size_t>(child.accepted);
                 k < child.col_ids.size(); ++k) {
                appendUnique(child.col_ids[k], front.col_ids, col_position_);
            }
        }
        front.matrix.assign(
            front.row_ids.size() * front.col_ids.size(), 0.0);
        assembleInput(node, row_position_, col_position_, front);
        assembleChildren(node, row_position_, col_position_, front);
        for (std::size_t i = 0; i < front.row_ids.size(); ++i) {
            row_position_[static_cast<std::size_t>(front.row_ids[i])] = -1;
        }
        for (std::size_t i = 0; i < front.col_ids.size(); ++i) {
            col_position_[static_cast<std::size_t>(front.col_ids[i])] = -1;
        }
        return front;
    }

    void assembleInput(
        int node,
        const std::vector<int>& row_position,
        const std::vector<int>& col_position,
        FrontData& front) const
    {
        if (input_rows_ == 0 || input_values_ == 0) {
            throw std::runtime_error("LU input buckets were released too early");
        }
        const int leading_dimension = static_cast<int>(front.row_ids.size());
        const int begin = input_owner_ptr_[static_cast<std::size_t>(node)];
        const int end = input_owner_ptr_[static_cast<std::size_t>(node + 1)];
        for (int q = begin; q < end; ++q) {
            const int entry = input_entry_indices_[static_cast<std::size_t>(q)];
            const int original_row =
                (*input_rows_)[static_cast<std::size_t>(entry)];
            const int original_col =
                input_entry_columns_[static_cast<std::size_t>(q)];
            const int row = row_position[static_cast<std::size_t>(original_row)];
            const int col = col_position[static_cast<std::size_t>(original_col)];
            if (row < 0 || col < 0) {
                throw std::runtime_error(
                    "symbolic envelope omitted an original unsymmetric entry");
            }
            front.matrix[static_cast<std::size_t>(
                row + col * leading_dimension)] +=
                    (*input_values_)[static_cast<std::size_t>(entry)];
        }
    }

    void assembleChildren(
        int node,
        const std::vector<int>& row_position,
        const std::vector<int>& col_position,
        FrontData& front)
    {
        const int leading_dimension = static_cast<int>(front.row_ids.size());
        const std::vector<int>& children = children_[static_cast<std::size_t>(node)];
        for (std::size_t c = 0; c < children.size(); ++c) {
            FactorData& child = factors_[static_cast<std::size_t>(children[c])];
            if (child.compacted) {
                throw std::runtime_error(
                    "LU child contribution was released before parent assembly");
            }
            const int child_rows = static_cast<int>(child.row_ids.size());
            for (std::size_t col = static_cast<std::size_t>(child.accepted);
                 col < child.col_ids.size(); ++col) {
                const int parent_col =
                    col_position[static_cast<std::size_t>(child.col_ids[col])];
                for (std::size_t row = static_cast<std::size_t>(child.accepted);
                     row < child.row_ids.size(); ++row) {
                    const int parent_row =
                        row_position[static_cast<std::size_t>(child.row_ids[row])];
                    if (parent_row < 0 || parent_col < 0) {
                        throw std::runtime_error(
                            "symbolic envelope omitted an LU contribution");
                    }
                    front.matrix[static_cast<std::size_t>(
                        parent_row + parent_col * leading_dimension)] +=
                        child.matrix[row + col * static_cast<std::size_t>(child_rows)];
                }
            }
            compactFactor(child);
        }
    }

    static std::size_t frontStorageBytes(const FrontData& front)
    {
        return front.matrix.size() * sizeof(LuScalar) +
            (front.row_ids.size() + front.col_ids.size()) * sizeof(int);
    }

    void storeFactor(
        int node,
        FactorData&& factor)
    {
        statistics_.accepted_pivots +=
            static_cast<std::size_t>(factor.accepted);
        statistics_.delayed_columns +=
            static_cast<std::size_t>(factor.delayed);
        factors_[static_cast<std::size_t>(node)] = std::move(factor);
    }

    void processFrontBatch(std::vector<FrontData>& fronts)
    {
        std::vector<int> small_indices;
        std::vector<int> large_indices;
        small_indices.reserve(fronts.size());
        large_indices.reserve(fronts.size());
        for (std::size_t i = 0; i < fronts.size(); ++i) {
            const int width = fronts[i].candidate_count;
            if (width == 1) {
                ++statistics_.single_column_nodes;
                small_indices.push_back(static_cast<int>(i));
            } else if (width <= options_.batched_width_limit &&
                       estimatedTrailingUpdates(fronts[i]) <=
                           options_.batched_update_limit) {
                ++statistics_.small_medium_nodes;
                small_indices.push_back(static_cast<int>(i));
            } else {
                ++statistics_.large_front_nodes;
                large_indices.push_back(static_cast<int>(i));
            }
        }

        std::vector<GpuFrontResult> small_results(fronts.size());
        const std::chrono::steady_clock::time_point small_begin =
            std::chrono::steady_clock::now();
        factorSmallFrontBatch(
            fronts, small_indices, options_, small_results);
        statistics_.small_medium_factorization_milliseconds +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - small_begin).count();
        for (std::size_t i = 0; i < small_indices.size(); ++i) {
            const int index = small_indices[i];
            GpuFrontResult& result = small_results[static_cast<std::size_t>(index)];
            storeFactor(
                result.node,
                std::move(result.factor));
        }

        const std::size_t concurrency = static_cast<std::size_t>(
            options_.concurrent_fronts);
        for (std::size_t first = 0;
             first < large_indices.size(); first += concurrency) {
            const std::size_t end = std::min(
                large_indices.size(), first + concurrency);
            std::vector<std::future<GpuFrontResult> > futures;
            for (std::size_t i = first; i < end; ++i) {
                const int index = large_indices[i];
                futures.push_back(std::async(
                    std::launch::async,
                    [this, front = std::move(
                         fronts[static_cast<std::size_t>(index)])]() mutable {
                        return factorLargeFrontOnGpu(
                            std::move(front), options_);
                    }));
            }
            for (std::size_t i = first; i < end; ++i) {
                const int index = large_indices[i];
                GpuFrontResult result = futures[i - first].get();
                statistics_.large_front_factorization_milliseconds +=
                    result.milliseconds;
                storeFactor(
                    result.node,
                    std::move(result.factor));
            }
        }
    }

    void processWave(
        const std::vector<int>& nodes,
        const UnsymmetricSymbolicResult& symbolic)
    {
        std::vector<FrontData> fronts;
        fronts.reserve(std::min(
            nodes.size(), static_cast<std::size_t>(options_.max_batch_nodes)));
        std::size_t batch_bytes = 0;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            const std::chrono::steady_clock::time_point assembly_begin =
                std::chrono::steady_clock::now();
            FrontData front = prepareFront(nodes[i], symbolic);
            statistics_.front_assembly_milliseconds +=
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - assembly_begin).count();
            const std::size_t bytes = frontStorageBytes(front);
            if (!fronts.empty() &&
                (fronts.size() >= options_.max_batch_nodes ||
                 batch_bytes + bytes > options_.max_batch_bytes)) {
                statistics_.maximum_batch_nodes = std::max(
                    statistics_.maximum_batch_nodes, fronts.size());
                statistics_.maximum_batch_bytes = std::max(
                    statistics_.maximum_batch_bytes, batch_bytes);
                processFrontBatch(fronts);
                fronts.clear();
                batch_bytes = 0;
            }
            batch_bytes += bytes;
            fronts.push_back(std::move(front));
        }
        if (!fronts.empty()) {
            statistics_.maximum_batch_nodes = std::max(
                statistics_.maximum_batch_nodes, fronts.size());
            statistics_.maximum_batch_bytes = std::max(
                statistics_.maximum_batch_bytes, batch_bytes);
            processFrontBatch(fronts);
        }
    }

    void forwardSolve(std::vector<LuScalar>& work) const
    {
        for (std::size_t node = 0; node < factors_.size(); ++node) {
            const FactorData& factor = factors_[node];
            const int rows = static_cast<int>(factor.row_ids.size());
            for (int k = 0; k < factor.accepted; ++k) {
                const LuScalar pivot_rhs =
                    work[static_cast<std::size_t>(factor.row_ids[k])];
                for (int row = k + 1; row < rows; ++row) {
                    const std::size_t position = static_cast<std::size_t>(
                        row + k * rows);
                    const LuScalar lower = factor.compacted
                        ? factor.factor_columns[position]
                        : factor.matrix[position];
                    work[static_cast<std::size_t>(factor.row_ids[row])] -=
                        lower *
                        pivot_rhs;
                }
            }
        }
    }

    std::vector<double> backwardSolve(const std::vector<LuScalar>& work) const
    {
        std::vector<LuScalar> solution(static_cast<std::size_t>(n_), 0.0);
        for (std::size_t reverse = factors_.size(); reverse-- > 0;) {
            const FactorData& factor = factors_[reverse];
            const int rows = static_cast<int>(factor.row_ids.size());
            const int cols = static_cast<int>(factor.col_ids.size());
            const int accepted = factor.accepted;
            for (int k = factor.accepted; k-- > 0;) {
                LuScalar value = work[static_cast<std::size_t>(factor.row_ids[k])];
                for (int col = k + 1; col < cols; ++col) {
                    const LuScalar upper = !factor.compacted
                        ? factor.matrix[static_cast<std::size_t>(
                              k + col * rows)]
                        : col < accepted
                            ? factor.factor_columns[static_cast<std::size_t>(
                                  k + col * rows)]
                            : factor.upper_trailing[static_cast<std::size_t>(
                                  k + (col - accepted) * accepted)];
                    value -= upper *
                        solution[static_cast<std::size_t>(factor.col_ids[col])];
                }
                const std::size_t diagonal = static_cast<std::size_t>(
                    k + k * rows);
                value /= factor.compacted
                    ? factor.factor_columns[diagonal]
                    : factor.matrix[diagonal];
                solution[static_cast<std::size_t>(factor.col_ids[k])] = value;
            }
        }
        return std::vector<double>(solution.begin(), solution.end());
    }

    GpuLuOptions options_;
    int n_;
    bool complete_;
    GpuLuStatistics statistics_;
    std::string diagnostic_;
    std::vector<FactorData> factors_;
    const std::vector<int>* input_rows_ = 0;
    const std::vector<double>* input_values_ = 0;
    std::vector<int> input_owner_ptr_;
    std::vector<int> input_entry_indices_;
    std::vector<int> input_entry_columns_;
    std::vector<std::vector<int> > children_;
    std::vector<int> row_position_;
    std::vector<int> col_position_;
};

GpuSupernodalLuFactor::GpuSupernodalLuFactor(const GpuLuOptions& options)
    : impl_(new Impl(options))
{
}

GpuSupernodalLuFactor::~GpuSupernodalLuFactor() = default;
GpuSupernodalLuFactor::GpuSupernodalLuFactor(
    GpuSupernodalLuFactor&& other) noexcept = default;
GpuSupernodalLuFactor& GpuSupernodalLuFactor::operator=(
    GpuSupernodalLuFactor&& other) noexcept = default;

GpuLuStatistics GpuSupernodalLuFactor::factorize(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<double>& values,
    const UnsymmetricSymbolicResult& symbolic)
{
    return impl_->factorize(n, col_ptr, row_indices, values, symbolic);
}

std::vector<double> GpuSupernodalLuFactor::solve(
    const std::vector<double>& reordered_rhs) const
{
    return impl_->solve(reordered_rhs);
}

bool GpuSupernodalLuFactor::complete() const { return impl_->complete(); }
const GpuLuStatistics& GpuSupernodalLuFactor::statistics() const
{
    return impl_->statistics();
}
const std::string& GpuSupernodalLuFactor::diagnostic() const
{
    return impl_->diagnostic();
}
