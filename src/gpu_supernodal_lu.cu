#include "gpu_supernodal_lu.hpp"

#include "gpu_supernodal_lu_kernels.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <future>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

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

struct MatrixEntry {
    int row;
    int col;
    float value;
};

struct FrontData {
    int node = -1;
    int candidate_count = 0;
    std::vector<int> row_ids;
    std::vector<int> col_ids;
    std::vector<float> matrix;
};

struct FactorData {
    int candidate_count = 0;
    int accepted = 0;
    int delayed = 0;
    std::vector<int> row_ids;
    std::vector<int> col_ids;
    std::vector<float> matrix;
};

struct GpuFrontResult {
    FactorData factor;
    bool large = false;
    float milliseconds = 0.0f;
};

void validateCsc(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values)
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

float frontScale(const FrontData& front)
{
    float scale = 0.0f;
    for (std::size_t i = 0; i < front.matrix.size(); ++i) {
        scale = std::max(scale, std::fabs(front.matrix[i]));
    }
    return scale;
}

void copyToDeviceAsync(
    DeviceArray<float>& device_matrix,
    DeviceArray<int>& device_rows,
    DeviceArray<int>& device_cols,
    const FrontData& front,
    cudaStream_t stream)
{
    checkCuda(cudaMemcpyAsync(
        device_matrix.get(), front.matrix.data(),
        front.matrix.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
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
    const DeviceArray<float>& device_matrix,
    const DeviceArray<int>& device_rows,
    const DeviceArray<int>& device_cols,
    cudaStream_t stream)
{
    checkCuda(cudaMemcpyAsync(
        factor.matrix.data(), device_matrix.get(),
        factor.matrix.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
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

void runSmallFront(
    DeviceArray<float>& matrix,
    DeviceArray<int>& rows,
    DeviceArray<int>& cols,
    int row_count,
    int col_count,
    int candidate_count,
    float threshold,
    float zero_tolerance,
    int& accepted,
    int& delayed,
    cudaStream_t stream)
{
    DeviceArray<int> output(2);
    unsymmetric_lu_kernels::factorSmallFront<<<1, 256, 0, stream>>>(
        matrix.get(), row_count, row_count, col_count, candidate_count,
        rows.get(), cols.get(), threshold, zero_tolerance,
        output.get(), output.get() + 1);
    checkCuda(cudaGetLastError(), "launch small/medium LU kernel");
    int host_output[2] = {0, 0};
    checkCuda(cudaMemcpyAsync(
        host_output, output.get(), sizeof(host_output),
        cudaMemcpyDeviceToHost, stream), "copy small LU result");
    checkCuda(cudaStreamSynchronize(stream), "synchronize small LU front");
    accepted = host_output[0];
    delayed = host_output[1];
}

int selectLargePivot(
    const DeviceArray<float>& matrix,
    int row_count,
    int candidate_count,
    int step,
    float threshold,
    float zero_tolerance,
    DeviceArray<int>& device_status,
    cudaStream_t stream)
{
    unsymmetric_lu_kernels::selectPivot<<<1, 1, 0, stream>>>(
        matrix.get(), row_count, row_count, candidate_count, step,
        threshold, zero_tolerance, device_status.get());
    checkCuda(cudaGetLastError(), "launch large-front pivot selection");
    int pivot = -1;
    checkCuda(cudaMemcpyAsync(
        &pivot, device_status.get(), sizeof(int), cudaMemcpyDeviceToHost,
        stream), "copy selected pivot row");
    checkCuda(cudaStreamSynchronize(stream), "synchronize pivot selection");
    return pivot;
}

int replaceLargeColumn(
    DeviceArray<float>& matrix,
    DeviceArray<int>& cols,
    int row_count,
    int candidate_count,
    int step,
    int& active_end,
    float threshold,
    float zero_tolerance,
    DeviceArray<int>& device_status,
    cudaStream_t stream)
{
    unsymmetric_lu_kernels::findReplacement<<<1, 1, 0, stream>>>(
        matrix.get(), row_count, row_count, candidate_count, step, active_end,
        threshold, zero_tolerance, device_status.get(), device_status.get() + 1);
    checkCuda(cudaGetLastError(), "launch delayed-column search");
    int status[2] = {-1, step};
    checkCuda(cudaMemcpyAsync(
        status, device_status.get(), sizeof(status), cudaMemcpyDeviceToHost,
        stream), "copy delayed-column search result");
    checkCuda(cudaStreamSynchronize(stream), "synchronize delayed-column search");
    active_end = status[1];
    if (status[0] >= 0) {
        const int blocks = std::max(1, (row_count + 255) / 256);
        unsymmetric_lu_kernels::swapColumns<<<blocks, 256, 0, stream>>>(
            matrix.get(), row_count, row_count, step, status[0], cols.get());
        checkCuda(cudaGetLastError(), "launch delayed-column swap");
    }
    return status[0];
}

void applyLargePivot(
    DeviceArray<float>& matrix,
    DeviceArray<int>& rows,
    int row_count,
    int col_count,
    int step,
    int pivot,
    int panel_end,
    cudaStream_t stream)
{
    if (pivot != step) {
        const int blocks = std::max(1, (col_count + 255) / 256);
        unsymmetric_lu_kernels::swapRows<<<blocks, 256, 0, stream>>>(
            matrix.get(), row_count, col_count, step, pivot, rows.get());
        checkCuda(cudaGetLastError(), "launch LU row swap");
    }
    const int row_blocks = std::max(1, (row_count - step + 255) / 256);
    unsymmetric_lu_kernels::dividePivotColumn<<<row_blocks, 256, 0, stream>>>(
        matrix.get(), row_count, row_count, step);
    checkCuda(cudaGetLastError(), "launch LU column division");

    if (step + 1 < panel_end && step + 1 < row_count) {
        const dim3 threads(16, 8);
        const dim3 blocks(
            (row_count - step - 1 + threads.x - 1) / threads.x,
            (panel_end - step - 1 + threads.y - 1) / threads.y);
        unsymmetric_lu_kernels::updatePanel<<<blocks, threads, 0, stream>>>(
            matrix.get(), row_count, row_count, step + 1, panel_end, step);
        checkCuda(cudaGetLastError(), "launch LU panel update");
    }
}

void flushLargePanel(
    DeviceArray<float>& matrix,
    int row_count,
    int col_count,
    int panel_begin,
    int panel_end,
    cublasHandle_t cublas)
{
    const int width = panel_end - panel_begin;
    const int right_cols = col_count - panel_end;
    if (width <= 0 || right_cols <= 0) {
        return;
    }
    const float one = 1.0f;
    checkCublas(cublasStrsm(
        cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N, CUBLAS_DIAG_UNIT, width, right_cols, &one,
        matrix.get() + panel_begin + panel_begin * row_count, row_count,
        matrix.get() + panel_begin + panel_end * row_count, row_count),
        "cuBLAS LU panel TRSM");

    const int trailing_rows = row_count - panel_end;
    if (trailing_rows <= 0) {
        return;
    }
    const float minus_one = -1.0f;
    checkCublas(cublasSgemm(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        trailing_rows, right_cols, width, &minus_one,
        matrix.get() + panel_end + panel_begin * row_count, row_count,
        matrix.get() + panel_begin + panel_end * row_count, row_count,
        &one, matrix.get() + panel_end + panel_end * row_count, row_count),
        "cuBLAS LU trailing GEMM");
}

void runLargeFront(
    DeviceArray<float>& matrix,
    DeviceArray<int>& rows,
    DeviceArray<int>& cols,
    int row_count,
    int col_count,
    int candidate_count,
    const GpuLuOptions& options,
    float zero_tolerance,
    int& accepted,
    int& delayed,
    cudaStream_t stream)
{
    cublasHandle_t cublas = 0;
    checkCublas(cublasCreate(&cublas), "create LU cuBLAS handle");
    try {
        checkCublas(cublasSetStream(cublas, stream), "set LU cuBLAS stream");
        DeviceArray<int> status(2);
        int step = 0;
        int active_end = candidate_count;
        while (step < active_end) {
            const int panel_begin = step;
            int panel_end = std::min(
                panel_begin + options.panel_size, active_end);
            while (step < panel_end) {
                const int pivot = selectLargePivot(
                    matrix, row_count, candidate_count, step,
                    options.threshold_pivoting, zero_tolerance, status, stream);
                if (pivot < 0) {
                    if (step != panel_begin) {
                        break;
                    }
                    const int replacement = replaceLargeColumn(
                        matrix, cols, row_count, candidate_count, step,
                        active_end, options.threshold_pivoting, zero_tolerance,
                        status, stream);
                    if (replacement < 0) {
                        break;
                    }
                    panel_end = std::min(
                        panel_begin + options.panel_size, active_end);
                    continue;
                }
                applyLargePivot(
                    matrix, rows, row_count, col_count,
                    step, pivot, panel_end, stream);
                ++step;
            }
            flushLargePanel(
                matrix, row_count, col_count, panel_begin, step, cublas);
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

GpuFrontResult factorFrontOnGpu(FrontData front, const GpuLuOptions& options)
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
        DeviceArray<float> device_matrix(front.matrix.size());
        DeviceArray<int> device_rows(front.row_ids.size());
        DeviceArray<int> device_cols(front.col_ids.size());
        copyToDeviceAsync(
            device_matrix, device_rows, device_cols, front, stream);

        GpuFrontResult result;
        result.large = front.candidate_count > options.batched_width_limit;
        result.factor.candidate_count = front.candidate_count;
        const float zero_tolerance = std::max(
            options.absolute_pivot_tolerance,
            options.relative_pivot_tolerance * frontScale(front));
        if (result.large) {
            runLargeFront(
                device_matrix, device_rows, device_cols, row_count, col_count,
                front.candidate_count, options, zero_tolerance,
                result.factor.accepted, result.factor.delayed, stream);
        } else {
            runSmallFront(
                device_matrix, device_rows, device_cols, row_count, col_count,
                front.candidate_count, options.threshold_pivoting,
                zero_tolerance, result.factor.accepted,
                result.factor.delayed, stream);
        }
        result.factor.row_ids.resize(front.row_ids.size());
        result.factor.col_ids.resize(front.col_ids.size());
        result.factor.matrix.resize(front.matrix.size());
        copyFromDeviceAsync(
            result.factor, device_matrix, device_rows, device_cols, stream);
        checkCuda(cudaStreamSynchronize(stream), "download factored LU front");
        result.milliseconds = std::chrono::duration<float, std::milli>(
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
      panel_size(64),
      concurrent_fronts(4),
      threshold_pivoting(0.1f),
      absolute_pivot_tolerance(1.0e-20f),
      relative_pivot_tolerance(1.0e-7f)
{
}

GpuLuStatistics::GpuLuStatistics()
    : single_column_nodes(0), small_medium_nodes(0), large_panel_nodes(0),
      accepted_pivots(0), delayed_columns(0), unresolved_root_columns(0),
      tree_waves(0), concurrent_fronts(0),
      front_assembly_milliseconds(0.0f),
      small_medium_factorization_milliseconds(0.0f),
      large_panel_factorization_milliseconds(0.0f),
      factorization_milliseconds(0.0f)
{
}

class GpuSupernodalLuFactor::Impl {
public:
    explicit Impl(const GpuLuOptions& options)
        : options_(options), n_(0), complete_(false)
    {
        if (options_.batched_width_limit < 1 || options_.panel_size < 1 ||
            options_.concurrent_fronts < 1 ||
            options_.threshold_pivoting <= 0.0f ||
            options_.threshold_pivoting > 1.0f) {
            throw std::invalid_argument("invalid general GPU LU options");
        }
    }

    GpuLuStatistics factorize(
        int n,
        const std::vector<int>& col_ptr,
        const std::vector<int>& row_indices,
        const std::vector<float>& values,
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
            std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - factor_begin).count();
        return statistics_;
    }

    std::vector<float> solve(const std::vector<float>& rhs) const
    {
        if (!complete_) {
            throw std::runtime_error("cannot solve with an incomplete general LU");
        }
        if (rhs.size() != static_cast<std::size_t>(n_)) {
            throw std::invalid_argument("general LU right-hand side size mismatch");
        }
        std::vector<float> work = rhs;
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
        input_buckets_.assign(
            symbolic.supernode_parent.size(), std::vector<MatrixEntry>());
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
        const std::vector<float>& values,
        const UnsymmetricSymbolicResult& symbolic)
    {
        std::vector<int> column_node(static_cast<std::size_t>(n_), -1);
        for (std::size_t node = 0; node < factors_.size(); ++node) {
            const int begin = symbolic.supernode_ptr[node];
            const int end = symbolic.supernode_ptr[node + 1];
            for (int col = begin; col < end; ++col) {
                column_node[static_cast<std::size_t>(col)] =
                    static_cast<int>(node);
            }
        }
        for (int col = 0; col < n_; ++col) {
            for (int p = col_ptr[static_cast<std::size_t>(col)];
                 p < col_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
                const int row = row_indices[static_cast<std::size_t>(p)];
                const int owner_col = std::min(row, col);
                const int owner = column_node[static_cast<std::size_t>(owner_col)];
                input_buckets_[static_cast<std::size_t>(owner)].push_back(
                    MatrixEntry{row, col, values[static_cast<std::size_t>(p)]});
            }
        }
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
        int id, std::vector<int>& ids, std::unordered_map<int, int>& positions)
    {
        if (positions.find(id) == positions.end()) {
            positions[id] = static_cast<int>(ids.size());
            ids.push_back(id);
        }
    }

    static void appendCandidate(
        int id, std::vector<int>& ids, std::unordered_map<int, int>& positions)
    {
        if (positions.find(id) != positions.end()) {
            throw std::runtime_error("duplicate fully-summed LU candidate");
        }
        positions[id] = static_cast<int>(ids.size());
        ids.push_back(id);
    }

    FrontData prepareFront(
        int node,
        const UnsymmetricSymbolicResult& symbolic) const
    {
        FrontData front;
        front.node = node;
        std::unordered_map<int, int> row_position;
        std::unordered_map<int, int> col_position;
        const int owned_begin = symbolic.supernode_ptr[static_cast<std::size_t>(node)];
        const int owned_end = symbolic.supernode_ptr[static_cast<std::size_t>(node + 1)];
        for (int id = owned_begin; id < owned_end; ++id) {
            appendCandidate(id, front.row_ids, row_position);
            appendCandidate(id, front.col_ids, col_position);
        }
        for (std::size_t c = 0; c < children_[static_cast<std::size_t>(node)].size(); ++c) {
            const FactorData& child = factors_[static_cast<std::size_t>(
                children_[static_cast<std::size_t>(node)][c])];
            for (int k = 0; k < child.delayed; ++k) {
                appendCandidate(
                    child.row_ids[static_cast<std::size_t>(child.accepted + k)],
                    front.row_ids, row_position);
                appendCandidate(
                    child.col_ids[static_cast<std::size_t>(child.accepted + k)],
                    front.col_ids, col_position);
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
            appendUnique(id, front.row_ids, row_position);
            appendUnique(id, front.col_ids, col_position);
        }
        for (std::size_t c = 0; c < children_[static_cast<std::size_t>(node)].size(); ++c) {
            const FactorData& child = factors_[static_cast<std::size_t>(
                children_[static_cast<std::size_t>(node)][c])];
            for (std::size_t k = static_cast<std::size_t>(child.accepted);
                 k < child.row_ids.size(); ++k) {
                appendUnique(child.row_ids[k], front.row_ids, row_position);
            }
            for (std::size_t k = static_cast<std::size_t>(child.accepted);
                 k < child.col_ids.size(); ++k) {
                appendUnique(child.col_ids[k], front.col_ids, col_position);
            }
        }
        front.matrix.assign(
            front.row_ids.size() * front.col_ids.size(), 0.0f);
        assembleInput(node, row_position, col_position, front);
        assembleChildren(node, row_position, col_position, front);
        return front;
    }

    void assembleInput(
        int node,
        const std::unordered_map<int, int>& row_position,
        const std::unordered_map<int, int>& col_position,
        FrontData& front) const
    {
        const int leading_dimension = static_cast<int>(front.row_ids.size());
        const std::vector<MatrixEntry>& entries =
            input_buckets_[static_cast<std::size_t>(node)];
        for (std::size_t p = 0; p < entries.size(); ++p) {
            const std::unordered_map<int, int>::const_iterator row =
                row_position.find(entries[p].row);
            const std::unordered_map<int, int>::const_iterator col =
                col_position.find(entries[p].col);
            if (row == row_position.end() || col == col_position.end()) {
                throw std::runtime_error(
                    "symbolic envelope omitted an original unsymmetric entry");
            }
            front.matrix[static_cast<std::size_t>(
                row->second + col->second * leading_dimension)] += entries[p].value;
        }
    }

    void assembleChildren(
        int node,
        const std::unordered_map<int, int>& row_position,
        const std::unordered_map<int, int>& col_position,
        FrontData& front) const
    {
        const int leading_dimension = static_cast<int>(front.row_ids.size());
        const std::vector<int>& children = children_[static_cast<std::size_t>(node)];
        for (std::size_t c = 0; c < children.size(); ++c) {
            const FactorData& child = factors_[static_cast<std::size_t>(children[c])];
            const int child_rows = static_cast<int>(child.row_ids.size());
            for (std::size_t col = static_cast<std::size_t>(child.accepted);
                 col < child.col_ids.size(); ++col) {
                const int parent_col = col_position.at(child.col_ids[col]);
                for (std::size_t row = static_cast<std::size_t>(child.accepted);
                     row < child.row_ids.size(); ++row) {
                    const int parent_row = row_position.at(child.row_ids[row]);
                    front.matrix[static_cast<std::size_t>(
                        parent_row + parent_col * leading_dimension)] +=
                        child.matrix[row + col * static_cast<std::size_t>(child_rows)];
                }
            }
        }
    }

    void processWave(
        const std::vector<int>& nodes,
        const UnsymmetricSymbolicResult& symbolic)
    {
        std::vector<FrontData> fronts;
        fronts.reserve(nodes.size());
        const std::chrono::steady_clock::time_point assembly_begin =
            std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            fronts.push_back(prepareFront(nodes[i], symbolic));
            const int width = fronts.back().candidate_count;
            if (width == 1) {
                ++statistics_.single_column_nodes;
            } else if (width <= options_.batched_width_limit) {
                ++statistics_.small_medium_nodes;
            } else {
                ++statistics_.large_panel_nodes;
            }
        }
        statistics_.front_assembly_milliseconds +=
            std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - assembly_begin).count();

        const std::size_t concurrency = static_cast<std::size_t>(
            options_.concurrent_fronts);
        for (std::size_t first = 0; first < nodes.size(); first += concurrency) {
            const std::size_t end = std::min(nodes.size(), first + concurrency);
            std::vector<std::future<GpuFrontResult> > futures;
            for (std::size_t i = first; i < end; ++i) {
                futures.push_back(std::async(
                    std::launch::async,
                    [this, front = std::move(fronts[i])]() mutable {
                        return factorFrontOnGpu(std::move(front), options_);
                    }));
            }
            for (std::size_t i = first; i < end; ++i) {
                GpuFrontResult result = futures[i - first].get();
                statistics_.accepted_pivots +=
                    static_cast<std::size_t>(result.factor.accepted);
                statistics_.delayed_columns +=
                    static_cast<std::size_t>(result.factor.delayed);
                if (result.large) {
                    statistics_.large_panel_factorization_milliseconds +=
                        result.milliseconds;
                } else {
                    statistics_.small_medium_factorization_milliseconds +=
                        result.milliseconds;
                }
                factors_[static_cast<std::size_t>(nodes[i])] =
                    std::move(result.factor);
            }
        }
    }

    void forwardSolve(std::vector<float>& work) const
    {
        for (std::size_t node = 0; node < factors_.size(); ++node) {
            const FactorData& factor = factors_[node];
            const int rows = static_cast<int>(factor.row_ids.size());
            for (int k = 0; k < factor.accepted; ++k) {
                const float pivot_rhs =
                    work[static_cast<std::size_t>(factor.row_ids[k])];
                for (int row = k + 1; row < rows; ++row) {
                    work[static_cast<std::size_t>(factor.row_ids[row])] -=
                        factor.matrix[static_cast<std::size_t>(row + k * rows)] *
                        pivot_rhs;
                }
            }
        }
    }

    std::vector<float> backwardSolve(const std::vector<float>& work) const
    {
        std::vector<float> solution(static_cast<std::size_t>(n_), 0.0f);
        for (std::size_t reverse = factors_.size(); reverse-- > 0;) {
            const FactorData& factor = factors_[reverse];
            const int rows = static_cast<int>(factor.row_ids.size());
            const int cols = static_cast<int>(factor.col_ids.size());
            for (int k = factor.accepted; k-- > 0;) {
                float value = work[static_cast<std::size_t>(factor.row_ids[k])];
                for (int col = k + 1; col < cols; ++col) {
                    value -= factor.matrix[static_cast<std::size_t>(k + col * rows)] *
                        solution[static_cast<std::size_t>(factor.col_ids[col])];
                }
                value /= factor.matrix[static_cast<std::size_t>(k + k * rows)];
                solution[static_cast<std::size_t>(factor.col_ids[k])] = value;
            }
        }
        return solution;
    }

    GpuLuOptions options_;
    int n_;
    bool complete_;
    GpuLuStatistics statistics_;
    std::string diagnostic_;
    std::vector<FactorData> factors_;
    std::vector<std::vector<MatrixEntry> > input_buckets_;
    std::vector<std::vector<int> > children_;
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
    const std::vector<float>& values,
    const UnsymmetricSymbolicResult& symbolic)
{
    return impl_->factorize(n, col_ptr, row_indices, values, symbolic);
}

std::vector<float> GpuSupernodalLuFactor::solve(
    const std::vector<float>& reordered_rhs) const
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
