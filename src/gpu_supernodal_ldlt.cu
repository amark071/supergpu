#include "gpu_supernodal_ldlt.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
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
        std::ostringstream message;
        message << operation << ": cuBLAS status " << static_cast<int>(status);
        throw std::runtime_error(message.str());
    }
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : data_(0), size_(0) {}
    explicit DeviceBuffer(std::size_t size) : data_(0), size_(0) { allocate(size); }
    ~DeviceBuffer() { reset(); }

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_)
    {
        other.data_ = 0;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other) {
            reset();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = 0;
            other.size_ = 0;
        }
        return *this;
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(std::size_t size)
    {
        reset();
        if (size != 0) {
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(T)),
                      "cudaMalloc");
            size_ = size;
        }
    }

    void reset() noexcept
    {
        if (data_ != 0) {
            cudaFree(data_);
        }
        data_ = 0;
        size_ = 0;
    }

    T* get() { return data_; }
    const T* get() const { return data_; }
    std::size_t size() const { return size_; }

private:
    T* data_;
    std::size_t size_;
};

struct DeviceFrontDescriptor {
    float* values;
    int* ids;
    int* pivot_size;
    int order;
    int candidate_count;
    float gamma;
    float absolute_tolerance;
    float relative_tolerance;
    float two_by_two_tolerance;
    int accepted;
    int delayed;
    int one_by_one;
    int two_by_two;
};

struct DeviceMaxPair {
    float value;
    int index;
};

struct DeviceLargePanelState {
    int step;
    int active_end;
    int panel_columns;
    int one_by_one;
    int two_by_two;
    int delay_current;
};

__device__ float absf(float value)
{
    return value < 0.0f ? -value : value;
}

__device__ float lowerValue(const float* matrix, int ld, int row, int col)
{
    const int r = row >= col ? row : col;
    const int c = row >= col ? col : row;
    return matrix[r + c * ld];
}

__device__ int binaryFind(
    const int* sorted_ids,
    const int* sorted_positions,
    int count,
    int id)
{
    int left = 0;
    int right = count;
    while (left < right) {
        const int middle = left + (right - left) / 2;
        const int candidate = sorted_ids[middle];
        if (candidate < id) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    return left < count && sorted_ids[left] == id ? sorted_positions[left] : -1;
}

__global__ void scatterBaseEntriesKernel(
    float* front,
    int ld,
    const int* sorted_ids,
    const int* sorted_positions,
    int id_count,
    const int* rows,
    const int* cols,
    const float* values,
    int begin,
    int end)
{
    const int entry = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (entry >= end) {
        return;
    }
    const int local_row = binaryFind(
        sorted_ids, sorted_positions, id_count, rows[entry]);
    const int local_col = binaryFind(
        sorted_ids, sorted_positions, id_count, cols[entry]);
    if (local_row < 0 || local_col < 0) {
        return;
    }
    const float value = values[entry];
    atomicAdd(front + local_row + local_col * ld, value);
    if (local_row != local_col) {
        atomicAdd(front + local_col + local_row * ld, value);
    }
}

__global__ void extendAddKernel(
    float* parent,
    int parent_ld,
    const int* sorted_parent_ids,
    const int* sorted_parent_positions,
    int parent_order,
    const float* child,
    const int* child_ids,
    int child_order)
{
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = child_order * child_order;
    if (linear >= total) {
        return;
    }
    const int row = linear % child_order;
    const int col = linear / child_order;
    if (row < col) {
        return;
    }
    const int parent_row = binaryFind(
        sorted_parent_ids, sorted_parent_positions, parent_order, child_ids[row]);
    const int parent_col = binaryFind(
        sorted_parent_ids, sorted_parent_positions, parent_order, child_ids[col]);
    if (parent_row < 0 || parent_col < 0) {
        return;
    }
    const float value = child[row + col * child_order];
    atomicAdd(parent + parent_row + parent_col * parent_ld, value);
    if (parent_row != parent_col) {
        atomicAdd(parent + parent_col + parent_row * parent_ld, value);
    }
}

__device__ void restrictedSymmetricSwapBlock(
    float* matrix,
    int ld,
    int order,
    int step,
    int lhs,
    int rhs,
    int* ids)
{
    if (lhs == rhs) {
        __syncthreads();
        return;
    }

    for (int col = threadIdx.x; col < step; col += blockDim.x) {
        const float value = matrix[lhs + col * ld];
        matrix[lhs + col * ld] = matrix[rhs + col * ld];
        matrix[rhs + col * ld] = value;
    }
    __syncthreads();

    for (int row = step + threadIdx.x; row < order; row += blockDim.x) {
        const float value = matrix[row + lhs * ld];
        matrix[row + lhs * ld] = matrix[row + rhs * ld];
        matrix[row + rhs * ld] = value;
    }
    __syncthreads();

    for (int col = step + threadIdx.x; col < order; col += blockDim.x) {
        const float value = matrix[lhs + col * ld];
        matrix[lhs + col * ld] = matrix[rhs + col * ld];
        matrix[rhs + col * ld] = value;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const int id = ids[lhs];
        ids[lhs] = ids[rhs];
        ids[rhs] = id;
    }
    __syncthreads();
}

__global__ void factorSingleColumnKernel(DeviceFrontDescriptor* descriptors)
{
    DeviceFrontDescriptor& front = descriptors[blockIdx.x];
    float* matrix = front.values;
    const int order = front.order;

    __shared__ int accepted;
    __shared__ float diagonal;
    __shared__ float column_maximum[256];
    float local_maximum = 0.0f;
    for (int row = 1 + threadIdx.x; row < order; row += blockDim.x) {
        local_maximum = fmaxf(local_maximum, absf(matrix[row]));
    }
    column_maximum[threadIdx.x] = local_maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            column_maximum[threadIdx.x] = fmaxf(
                column_maximum[threadIdx.x],
                column_maximum[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        diagonal = matrix[0];
        const float diagonal_absolute = absf(diagonal);
        const float scale = fmaxf(diagonal_absolute, column_maximum[0]);
        const float tolerance = fmaxf(
            front.absolute_tolerance,
            front.relative_tolerance * scale);
        accepted = diagonal_absolute > tolerance &&
                   (column_maximum[0] <= tolerance ||
                    diagonal_absolute >= front.gamma * column_maximum[0])
            ? 1 : 0;
        front.pivot_size[0] = accepted ? 1 : 0;
    }
    __syncthreads();

    if (accepted) {
        for (int row = 1 + threadIdx.x; row < order; row += blockDim.x) {
            matrix[row] /= diagonal;
        }
        __syncthreads();

        const int remaining = order - 1;
        const int total = remaining * remaining;
        for (int linear = threadIdx.x; linear < total; linear += blockDim.x) {
            const int row = 1 + linear % remaining;
            const int col = 1 + linear / remaining;
            if (row >= col) {
                const float updated = matrix[row + col * order] -
                    matrix[row] * diagonal * matrix[col];
                matrix[row + col * order] = updated;
                if (row != col) {
                    matrix[col + row * order] = updated;
                }
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        front.accepted = accepted;
        front.delayed = 1 - accepted;
        front.one_by_one = accepted;
        front.two_by_two = 0;
    }
}

__global__ void factorSmallMediumKernel(DeviceFrontDescriptor* descriptors)
{
    DeviceFrontDescriptor& front = descriptors[blockIdx.x];
    float* matrix = front.values;
    const int order = front.order;

    __shared__ int step;
    __shared__ int active_end;
    __shared__ int decision;
    __shared__ int pivot_row;
    __shared__ int one_count;
    __shared__ int two_count;
    __shared__ float pivot_a;
    __shared__ float pivot_b;
    __shared__ float pivot_c;

    if (threadIdx.x == 0) {
        step = 0;
        active_end = front.candidate_count;
        one_count = 0;
        two_count = 0;
    }
    __syncthreads();

    while (true) {
        if (threadIdx.x == 0) {
            decision = 0;
            pivot_row = step;
            if (step < active_end) {
                const float diagonal = absf(matrix[step + step * order]);
                float lambda = 0.0f;
                int p = step;
                // 非 fully-summed 行不能作为主元，但必须参与列稳定性检查。
                for (int row = step + 1; row < order; ++row) {
                    const float candidate = absf(matrix[row + step * order]);
                    if (candidate > lambda) {
                        lambda = candidate;
                        p = row;
                    }
                }
                const float scale = fmaxf(diagonal, lambda);
                const float tolerance = fmaxf(
                    front.absolute_tolerance,
                    front.relative_tolerance * scale);

                if (lambda <= tolerance) {
                    decision = diagonal > tolerance ? 1 : 3;
                } else if (diagonal >= front.gamma * lambda) {
                    decision = 1;
                } else if (p >= active_end) {
                    // The BK row test requires a fully-summed pivot candidate.
                    decision = 3;
                } else {
                    float sigma = 0.0f;
                    for (int col = step; col < order; ++col) {
                        if (col != p) {
                            sigma = fmaxf(sigma, absf(lowerValue(matrix, order, p, col)));
                        }
                    }
                    if (sigma <= tolerance ||
                        diagonal >= front.gamma * lambda * lambda / sigma) {
                        decision = 1;
                    } else if (absf(matrix[p + p * order]) >= front.gamma * sigma) {
                        decision = 1;
                        pivot_row = p;
                    } else if (step + 1 < active_end) {
                        const float a = matrix[step + step * order];
                        const float b = matrix[p + step * order];
                        const float c = matrix[p + p * order];
                        const float determinant = a * c - b * b;
                        const float norm = fmaxf(absf(a) + absf(b), absf(b) + absf(c));
                        if (absf(determinant) >
                                front.two_by_two_tolerance * norm * norm &&
                            absf(determinant) > tolerance * tolerance) {
                            decision = 2;
                            pivot_row = p;
                            pivot_a = a;
                            pivot_b = b;
                            pivot_c = c;
                        } else {
                            decision = 3;
                        }
                    } else {
                        decision = 3;
                    }
                }
            }
        }
        __syncthreads();

        if (step >= active_end) {
            break;
        }

        if (decision == 3) {
            restrictedSymmetricSwapBlock(
                matrix, order, order, step, step, active_end - 1, front.ids);
            if (threadIdx.x == 0) {
                --active_end;
            }
            __syncthreads();
            continue;
        }

        if (decision == 1) {
            restrictedSymmetricSwapBlock(
                matrix, order, order, step, step, pivot_row, front.ids);
            if (threadIdx.x == 0) {
                pivot_a = matrix[step + step * order];
                front.pivot_size[step] = 1;
            }
            __syncthreads();

            for (int row = step + 1 + threadIdx.x; row < order; row += blockDim.x) {
                matrix[row + step * order] /= pivot_a;
            }
            __syncthreads();

            const int remaining = order - step - 1;
            const int total = remaining * remaining;
            for (int linear = threadIdx.x; linear < total; linear += blockDim.x) {
                const int row = step + 1 + linear % remaining;
                const int col = step + 1 + linear / remaining;
                if (row >= col) {
                    const float updated = matrix[row + col * order] -
                        matrix[row + step * order] * pivot_a *
                        matrix[col + step * order];
                    matrix[row + col * order] = updated;
                    if (row != col) {
                        matrix[col + row * order] = updated;
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                ++step;
                ++one_count;
            }
            __syncthreads();
            continue;
        }

        restrictedSymmetricSwapBlock(
            matrix, order, order, step, step + 1, pivot_row, front.ids);
        if (threadIdx.x == 0) {
            pivot_a = matrix[step + step * order];
            pivot_b = matrix[step + 1 + step * order];
            pivot_c = matrix[step + 1 + (step + 1) * order];
            front.pivot_size[step] = 2;
            front.pivot_size[step + 1] = 0;
        }
        __syncthreads();

        const float determinant = pivot_a * pivot_c - pivot_b * pivot_b;
        for (int row = step + 2 + threadIdx.x; row < order; row += blockDim.x) {
            const float first = matrix[row + step * order];
            const float second = matrix[row + (step + 1) * order];
            matrix[row + step * order] =
                (first * pivot_c - second * pivot_b) / determinant;
            matrix[row + (step + 1) * order] =
                (second * pivot_a - first * pivot_b) / determinant;
        }
        __syncthreads();

        const int remaining = order - step - 2;
        const int total = remaining * remaining;
        for (int linear = threadIdx.x; linear < total; linear += blockDim.x) {
            const int row = step + 2 + linear % remaining;
            const int col = step + 2 + linear / remaining;
            if (row >= col) {
                const float lr0 = matrix[row + step * order];
                const float lr1 = matrix[row + (step + 1) * order];
                const float lc0 = matrix[col + step * order];
                const float lc1 = matrix[col + (step + 1) * order];
                const float wr0 = lr0 * pivot_a + lr1 * pivot_b;
                const float wr1 = lr0 * pivot_b + lr1 * pivot_c;
                const float updated = matrix[row + col * order] -
                    wr0 * lc0 - wr1 * lc1;
                matrix[row + col * order] = updated;
                if (row != col) {
                    matrix[col + row * order] = updated;
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            step += 2;
            ++two_count;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        front.accepted = step;
        front.delayed = front.candidate_count - step;
        front.one_by_one = one_count;
        front.two_by_two = two_count;
    }
}

__global__ void extractContributionKernel(
    const float* front,
    int front_order,
    int accepted,
    float* contribution,
    int contribution_order)
{
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = contribution_order * contribution_order;
    if (linear < total) {
        const int row = linear % contribution_order;
        const int col = linear / contribution_order;
        contribution[linear] =
            front[accepted + row + (accepted + col) * front_order];
    }
}

__device__ float panelEntry(
    const float* matrix,
    const float* work,
    int ld,
    int panel_begin,
    int panel_columns,
    int row,
    int col)
{
    const int r = row >= col ? row : col;
    const int c = row >= col ? col : row;
    float value = matrix[r + c * ld];
    for (int q = 0; q < panel_columns; ++q) {
        value -= matrix[r + (panel_begin + q) * ld] * work[c + q * ld];
    }
    return value;
}

__device__ DeviceMaxPair blockPanelColumnMaximum(
    const float* matrix,
    const float* work,
    int ld,
    int panel_begin,
    int panel_columns,
    int step,
    float* shared_values,
    int* shared_indices)
{
    float best = 0.0f;
    int best_index = step;
    for (int row = step + 1 + threadIdx.x; row < ld; row += blockDim.x) {
        const float value = absf(panelEntry(
            matrix, work, ld, panel_begin, panel_columns, row, step));
        if (value > best) {
            best = value;
            best_index = row;
        }
    }
    shared_values[threadIdx.x] = best;
    shared_indices[threadIdx.x] = best_index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride &&
            shared_values[threadIdx.x + stride] > shared_values[threadIdx.x]) {
            shared_values[threadIdx.x] = shared_values[threadIdx.x + stride];
            shared_indices[threadIdx.x] = shared_indices[threadIdx.x + stride];
        }
        __syncthreads();
    }
    DeviceMaxPair result;
    result.value = shared_values[0];
    result.index = shared_indices[0];
    return result;
}

__device__ DeviceMaxPair blockPanelRowMaximum(
    const float* matrix,
    const float* work,
    int ld,
    int panel_begin,
    int panel_columns,
    int row,
    int step,
    float* shared_values,
    int* shared_indices)
{
    float best = 0.0f;
    int best_index = step;
    for (int col = step + threadIdx.x; col < ld; col += blockDim.x) {
        if (col != row) {
            const float value = absf(panelEntry(
                matrix, work, ld, panel_begin, panel_columns, row, col));
            if (value > best) {
                best = value;
                best_index = col;
            }
        }
    }
    shared_values[threadIdx.x] = best;
    shared_indices[threadIdx.x] = best_index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride &&
            shared_values[threadIdx.x + stride] > shared_values[threadIdx.x]) {
            shared_values[threadIdx.x] = shared_values[threadIdx.x + stride];
            shared_indices[threadIdx.x] = shared_indices[threadIdx.x + stride];
        }
        __syncthreads();
    }
    DeviceMaxPair result;
    result.value = shared_values[0];
    result.index = shared_indices[0];
    return result;
}

__device__ void swapLargePanelRowsAndColumns(
    float* matrix,
    float* work,
    int ld,
    int panel_columns,
    int step,
    int lhs,
    int rhs,
    int* ids)
{
    restrictedSymmetricSwapBlock(matrix, ld, ld, step, lhs, rhs, ids);
    if (lhs == rhs) {
        return;
    }
    for (int col = threadIdx.x; col < panel_columns; col += blockDim.x) {
        const float value = work[lhs + col * ld];
        work[lhs + col * ld] = work[rhs + col * ld];
        work[rhs + col * ld] = value;
    }
    __syncthreads();
}

__global__ void factorLargePanelControlKernel(
    float* matrix,
    float* work,
    int* ids,
    int* pivot_size,
    int order,
    int panel_begin,
    int initial_active_end,
    int delay_first,
    int panel_capacity,
    float gamma,
    float absolute_tolerance,
    float relative_tolerance,
    float two_by_two_tolerance,
    DeviceLargePanelState* output)
{
    __shared__ float reduction_values[256];
    __shared__ int reduction_indices[256];
    __shared__ int step;
    __shared__ int active_end;
    __shared__ int panel_columns;
    __shared__ int decision;
    __shared__ int pivot_row;
    __shared__ int one_by_one;
    __shared__ int two_by_two;
    __shared__ float diagonal_value;
    __shared__ float column_maximum;
    __shared__ float tolerance;
    __shared__ float pivot_a;
    __shared__ float pivot_b;
    __shared__ float pivot_c;

    if (threadIdx.x == 0) {
        step = panel_begin;
        active_end = initial_active_end;
        panel_columns = 0;
        one_by_one = 0;
        two_by_two = 0;
    }
    __syncthreads();

    if (delay_first != 0 && step < active_end) {
        swapLargePanelRowsAndColumns(
            matrix, work, order, 0, step, step, active_end - 1, ids);
        if (threadIdx.x == 0) {
            --active_end;
        }
        __syncthreads();
    }

    while (step < active_end && panel_columns < panel_capacity) {
        if (threadIdx.x == 0) {
            diagonal_value = panelEntry(
                matrix, work, order, panel_begin, panel_columns, step, step);
            decision = -1;
            pivot_row = step;
        }
        __syncthreads();

        const DeviceMaxPair column_max = blockPanelColumnMaximum(
            matrix, work, order, panel_begin, panel_columns, step,
            reduction_values, reduction_indices);
        if (threadIdx.x == 0) {
            column_maximum = column_max.value;
            pivot_row = column_max.index;
            const float diagonal = absf(diagonal_value);
            const float scale = fmaxf(diagonal, column_maximum);
            tolerance = fmaxf(absolute_tolerance, relative_tolerance * scale);
            if (column_maximum <= tolerance) {
                decision = diagonal > tolerance ? 1 : 0;
            } else if (diagonal >= gamma * column_maximum) {
                decision = 1;
                pivot_row = step;
            } else if (pivot_row >= active_end) {
                // A non-fully-summed row cannot participate in the BK row test.
                decision = 0;
            }
        }
        __syncthreads();

        if (decision < 0) {
            const DeviceMaxPair row_max = blockPanelRowMaximum(
                matrix, work, order, panel_begin, panel_columns,
                pivot_row, step, reduction_values, reduction_indices);
            if (threadIdx.x == 0) {
                const float sigma = row_max.value;
                const float diagonal = absf(diagonal_value);
                const float candidate_diagonal = absf(panelEntry(
                    matrix, work, order, panel_begin, panel_columns,
                    pivot_row, pivot_row));
                if (sigma <= tolerance ||
                    diagonal >= gamma * column_maximum * column_maximum / sigma) {
                    decision = 1;
                    pivot_row = step;
                } else if (candidate_diagonal >= gamma * sigma) {
                    decision = 1;
                } else if (step + 1 < active_end) {
                    pivot_a = diagonal_value;
                    pivot_b = panelEntry(
                        matrix, work, order, panel_begin, panel_columns,
                        pivot_row, step);
                    pivot_c = panelEntry(
                        matrix, work, order, panel_begin, panel_columns,
                        pivot_row, pivot_row);
                    const float determinant = pivot_a * pivot_c - pivot_b * pivot_b;
                    const float norm = fmaxf(
                        absf(pivot_a) + absf(pivot_b),
                        absf(pivot_b) + absf(pivot_c));
                    decision = absf(determinant) >
                                   two_by_two_tolerance * norm * norm &&
                               absf(determinant) > tolerance * tolerance
                        ? 2 : 0;
                } else {
                    decision = 0;
                }
            }
            __syncthreads();
        }

        if (decision == 0) {
            // Preserve delayed-pivot semantics: flush a non-empty panel first.
            if (panel_columns != 0) {
                break;
            }
            swapLargePanelRowsAndColumns(
                matrix, work, order, 0, step, step, active_end - 1, ids);
            if (threadIdx.x == 0) {
                --active_end;
            }
            __syncthreads();
            continue;
        }

        if (panel_columns + decision > panel_capacity) {
            break;
        }

        if (decision == 1) {
            swapLargePanelRowsAndColumns(
                matrix, work, order, panel_columns, step, step, pivot_row, ids);
            if (threadIdx.x == 0) {
                pivot_a = panelEntry(
                    matrix, work, order, panel_begin, panel_columns, step, step);
            }
            __syncthreads();

            for (int row = step + threadIdx.x; row < order; row += blockDim.x) {
                const float value = panelEntry(
                    matrix, work, order, panel_begin, panel_columns, row, step);
                if (row == step) {
                    matrix[row + step * order] = pivot_a;
                    work[row + panel_columns * order] = pivot_a;
                } else {
                    matrix[row + step * order] = value / pivot_a;
                    work[row + panel_columns * order] = value;
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                pivot_size[step] = 1;
                ++step;
                ++panel_columns;
                ++one_by_one;
            }
            __syncthreads();
            continue;
        }

        swapLargePanelRowsAndColumns(
            matrix, work, order, panel_columns, step, step + 1, pivot_row, ids);
        if (threadIdx.x == 0) {
            pivot_a = panelEntry(
                matrix, work, order, panel_begin, panel_columns, step, step);
            pivot_b = panelEntry(
                matrix, work, order, panel_begin, panel_columns, step + 1, step);
            pivot_c = panelEntry(
                matrix, work, order, panel_begin, panel_columns, step + 1, step + 1);
        }
        __syncthreads();
        const float determinant = pivot_a * pivot_c - pivot_b * pivot_b;
        for (int row = step + 2 + threadIdx.x; row < order; row += blockDim.x) {
            const float first = panelEntry(
                matrix, work, order, panel_begin, panel_columns, row, step);
            const float second = panelEntry(
                matrix, work, order, panel_begin, panel_columns, row, step + 1);
            matrix[row + step * order] =
                (first * pivot_c - second * pivot_b) / determinant;
            matrix[row + (step + 1) * order] =
                (second * pivot_a - first * pivot_b) / determinant;
            work[row + panel_columns * order] = first;
            work[row + (panel_columns + 1) * order] = second;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            matrix[step + step * order] = pivot_a;
            matrix[step + 1 + step * order] = pivot_b;
            matrix[step + 1 + (step + 1) * order] = pivot_c;
            work[step + panel_columns * order] = pivot_a;
            work[step + (panel_columns + 1) * order] = pivot_b;
            work[step + 1 + panel_columns * order] = pivot_b;
            work[step + 1 + (panel_columns + 1) * order] = pivot_c;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            pivot_size[step] = 2;
            pivot_size[step + 1] = 0;
            step += 2;
            panel_columns += 2;
            ++two_by_two;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output->step = step;
        output->active_end = active_end;
        output->panel_columns = panel_columns;
        output->one_by_one = one_by_one;
        output->two_by_two = two_by_two;
        output->delay_current =
            step < active_end && panel_columns != 0 && decision == 0 ? 1 : 0;
    }
}

__global__ void mirrorLowerToUpperKernel(float* matrix, int ld, int begin)
{
    const int remaining = ld - begin;
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = remaining * remaining;
    if (linear >= total) {
        return;
    }
    const int row = begin + linear % remaining;
    const int col = begin + linear / remaining;
    if (row > col) {
        matrix[col + row * ld] = matrix[row + col * ld];
    }
}

__global__ void forwardFactorKernel(
    float* rhs,
    const float* factor,
    const int* ids,
    const int* pivot_size,
    int order,
    int accepted)
{
    __shared__ float first;
    __shared__ float second;
    int step = 0;
    while (step < accepted) {
        const int block_size = pivot_size[step];
        if (threadIdx.x == 0) {
            first = rhs[ids[step]];
            second = block_size == 2 ? rhs[ids[step + 1]] : 0.0f;
        }
        __syncthreads();
        const int row_begin = step + block_size;
        for (int row = row_begin + threadIdx.x; row < order; row += blockDim.x) {
            float value = factor[row + step * order] * first;
            if (block_size == 2) {
                value += factor[row + (step + 1) * order] * second;
            }
            rhs[ids[row]] -= value;
        }
        __syncthreads();
        step += block_size;
    }
}

__global__ void diagonalSolveKernel(
    float* rhs,
    const float* factor,
    const int* ids,
    const int* pivot_size,
    int order,
    int accepted)
{
    for (int step = threadIdx.x; step < accepted; step += blockDim.x) {
        if (pivot_size[step] == 1) {
            rhs[ids[step]] /= factor[step + step * order];
        } else if (pivot_size[step] == 2) {
            const float a = factor[step + step * order];
            const float b = factor[step + 1 + step * order];
            const float c = factor[step + 1 + (step + 1) * order];
            const float determinant = a * c - b * b;
            const float first = rhs[ids[step]];
            const float second = rhs[ids[step + 1]];
            rhs[ids[step]] = (c * first - b * second) / determinant;
            rhs[ids[step + 1]] = (a * second - b * first) / determinant;
        }
    }
}

__global__ void backwardFactorKernel(
    float* rhs,
    const float* factor,
    const int* ids,
    const int* pivot_size,
    int order,
    int accepted)
{
    __shared__ float partial_first[256];
    __shared__ float partial_second[256];
    int last = accepted - 1;
    while (last >= 0) {
        int step = last;
        if (pivot_size[step] == 0) {
            --step;
        }
        const int block_size = pivot_size[step];
        float sum_first = 0.0f;
        float sum_second = 0.0f;
        for (int row = step + block_size + threadIdx.x;
             row < order;
             row += blockDim.x) {
            const float value = rhs[ids[row]];
            sum_first += factor[row + step * order] * value;
            if (block_size == 2) {
                sum_second += factor[row + (step + 1) * order] * value;
            }
        }
        partial_first[threadIdx.x] = sum_first;
        partial_second[threadIdx.x] = sum_second;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                partial_first[threadIdx.x] += partial_first[threadIdx.x + stride];
                partial_second[threadIdx.x] += partial_second[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            rhs[ids[step]] -= partial_first[0];
            if (block_size == 2) {
                rhs[ids[step + 1]] -= partial_second[0];
            }
        }
        __syncthreads();
        last = step - 1;
    }
}

} // namespace

GpuLdltOptions::GpuLdltOptions()
    : batched_width_limit(64),
      panel_size(64),
      max_batch_nodes(256),
      bunch_kaufman_gamma((1.0f + std::sqrt(17.0f)) / 8.0f),
      absolute_pivot_tolerance(0.0f),
      relative_pivot_tolerance(32.0f * std::numeric_limits<float>::epsilon()),
      two_by_two_tolerance(64.0f * std::numeric_limits<float>::epsilon()),
      symmetrize_input(true)
{
}

GpuLdltStatistics::GpuLdltStatistics()
    : single_column_nodes(0),
      batched_small_medium_nodes(0),
      large_panel_nodes(0),
      accepted_one_by_one_pivots(0),
      accepted_two_by_two_pivots(0),
      delayed_columns(0),
      unresolved_root_columns(0),
      tree_waves(0),
      factorization_milliseconds(0.0f),
      maximum_input_asymmetry(0.0f)
{
}

class GpuSupernodalLdltFactor::Impl {
public:
    explicit Impl(const GpuLdltOptions& options)
        : options_(options),
          n_(0),
          complete_(false),
          cublas_(0)
    {
        validateOptions();
        checkCublas(cublasCreate(&cublas_), "cublasCreate");
        // FP32 输入、输出和累加；禁止 SGEMM 自动降为 TF32 mantissa。
        checkCublas(cublasSetMathMode(cublas_, CUBLAS_PEDANTIC_MATH),
                    "cublasSetMathMode");
    }

    ~Impl()
    {
        if (cublas_ != 0) {
            cublasDestroy(cublas_);
        }
    }

    GpuLdltStatistics factorize(
        int n,
        const std::vector<int>& col_ptr,
        const std::vector<int>& row_indices,
        const std::vector<float>& values,
        const CholmodSymbolicResult& symbolic)
    {
        resetNumericState();
        validateInput(n, col_ptr, row_indices, values, symbolic);
        n_ = n;
        cudaEvent_t start = 0;
        cudaEvent_t stop = 0;
        checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
        try {
            checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
            checkCuda(cudaEventRecord(start), "cudaEventRecord(start)");

            prepareTree(symbolic);
            prepareBaseEntries(n, col_ptr, row_indices, values, symbolic);
            runTreeWaves(symbolic);

            checkCuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
            checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
            checkCuda(cudaEventElapsedTime(
                          &statistics_.factorization_milliseconds, start, stop),
                      "cudaEventElapsedTime");
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
        } catch (...) {
            if (stop != 0) {
                cudaEventDestroy(stop);
            }
            if (start != 0) {
                cudaEventDestroy(start);
            }
            throw;
        }

        complete_ = statistics_.unresolved_root_columns == 0 &&
                    acceptedVariableCount() == static_cast<std::size_t>(n_);
        if (complete_) {
            diagnostic_ = "factorization completed";
        } else {
            std::ostringstream message;
            message << "factorization incomplete: "
                    << statistics_.unresolved_root_columns
                    << " root candidate columns remain delayed; accepted "
                    << acceptedVariableCount() << " of " << n_;
            diagnostic_ = message.str();
        }
        for (std::size_t node = 0; node < nodes_.size(); ++node) {
            nodes_[node].contribution_values.reset();
            nodes_[node].contribution_ids.reset();
            nodes_[node].contribution_ids_host.clear();
        }
        return statistics_;
    }

    std::vector<float> solve(const std::vector<float>& rhs) const
    {
        if (!complete_) {
            throw std::logic_error(
                "cannot solve with an incomplete GPU LDLT factor: " + diagnostic_);
        }
        if (rhs.size() != static_cast<std::size_t>(n_)) {
            throw std::invalid_argument("right-hand side size does not match the factor");
        }

        DeviceBuffer<float> device_rhs(rhs.size());
        checkCuda(cudaMemcpy(
                      device_rhs.get(), rhs.data(), rhs.size() * sizeof(float),
                      cudaMemcpyHostToDevice),
                  "copy RHS to GPU");

        for (std::size_t i = 0; i < factorization_order_.size(); ++i) {
            const NodeData& node = nodes_[factorization_order_[i]];
            if (node.accepted == 0) {
                continue;
            }
            forwardFactorKernel<<<1, 256>>>(
                device_rhs.get(), node.factor_values.get(), node.factor_ids.get(),
                node.pivot_size.get(), node.order, node.accepted);
        }
        checkCuda(cudaGetLastError(), "launch forward solve kernels");

        for (std::size_t i = 0; i < factorization_order_.size(); ++i) {
            const NodeData& node = nodes_[factorization_order_[i]];
            if (node.accepted == 0) {
                continue;
            }
            diagonalSolveKernel<<<1, 256>>>(
                device_rhs.get(), node.factor_values.get(), node.factor_ids.get(),
                node.pivot_size.get(), node.order, node.accepted);
        }
        checkCuda(cudaGetLastError(), "launch diagonal solve kernels");

        for (std::size_t reverse = factorization_order_.size(); reverse > 0; --reverse) {
            const NodeData& node = nodes_[factorization_order_[reverse - 1]];
            if (node.accepted == 0) {
                continue;
            }
            backwardFactorKernel<<<1, 256>>>(
                device_rhs.get(), node.factor_values.get(), node.factor_ids.get(),
                node.pivot_size.get(), node.order, node.accepted);
        }
        checkCuda(cudaGetLastError(), "launch backward solve kernels");

        std::vector<float> solution(rhs.size());
        checkCuda(cudaMemcpy(
                      solution.data(), device_rhs.get(), rhs.size() * sizeof(float),
                      cudaMemcpyDeviceToHost),
                  "copy solution from GPU");
        return solution;
    }

    bool complete() const { return complete_; }
    const GpuLdltStatistics& statistics() const { return statistics_; }
    const std::string& diagnostic() const { return diagnostic_; }

private:
    struct SymmetricAccumulator {
        SymmetricAccumulator()
            : lower(0.0f), upper(0.0f), lower_count(0), upper_count(0) {}
        float lower;
        float upper;
        int lower_count;
        int upper_count;
    };

    struct BaseEntry {
        int row;
        int col;
        float value;
    };

    struct NodeData {
        NodeData()
            : parent(-1), order(0), accepted(0), delayed(0), contribution_order(0) {}

        int parent;
        std::vector<int> children;
        int order;
        int accepted;
        int delayed;
        DeviceBuffer<float> factor_values;
        DeviceBuffer<int> factor_ids;
        DeviceBuffer<int> pivot_size;
        int contribution_order;
        std::vector<int> contribution_ids_host;
        DeviceBuffer<float> contribution_values;
        DeviceBuffer<int> contribution_ids;
    };

    struct FrontWork {
        FrontWork()
            : node(-1), order(0), candidate_count(0), accepted(0), delayed(0),
              one_by_one(0), two_by_two(0), classification(GpuSupernodeClass::SingleColumn)
        {
        }

        int node;
        int order;
        int candidate_count;
        int accepted;
        int delayed;
        int one_by_one;
        int two_by_two;
        GpuSupernodeClass classification;
        std::vector<int> ids_host;
        DeviceBuffer<float> values;
        DeviceBuffer<int> ids;
        DeviceBuffer<int> pivot_size;
        DeviceBuffer<int> sorted_ids;
        DeviceBuffer<int> sorted_positions;
    };

    void validateOptions() const
    {
        if (options_.batched_width_limit < 2 ||
            options_.panel_size < 2 ||
            options_.max_batch_nodes < 1 ||
            !(options_.bunch_kaufman_gamma > 0.0f) ||
            options_.absolute_pivot_tolerance < 0.0f ||
            options_.relative_pivot_tolerance < 0.0f ||
            options_.two_by_two_tolerance < 0.0f) {
            throw std::invalid_argument("invalid GPU LDLT options");
        }
    }

    static void validateInput(
        int n,
        const std::vector<int>& col_ptr,
        const std::vector<int>& row_indices,
        const std::vector<float>& values,
        const CholmodSymbolicResult& symbolic)
    {
        if (n < 0 || col_ptr.size() != static_cast<std::size_t>(n + 1) ||
            col_ptr.empty() || col_ptr.front() != 0 ||
            col_ptr.back() != static_cast<int>(row_indices.size()) ||
            row_indices.size() != values.size()) {
            throw std::invalid_argument("invalid numeric CSC matrix");
        }
        for (int col = 0; col < n; ++col) {
            if (col_ptr[col] > col_ptr[col + 1]) {
                throw std::invalid_argument("CSC column pointers are not monotonic");
            }
            for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
                if (row_indices[p] < 0 || row_indices[p] >= n ||
                    !std::isfinite(values[p])) {
                    throw std::invalid_argument("invalid CSC row or non-finite value");
                }
            }
        }

        const std::size_t node_count = symbolic.supernode_parent.size();
        if (symbolic.supernode_ptr.size() != node_count + 1 ||
            symbolic.row_ptr.size() != node_count + 1 ||
            symbolic.supernode_ptr.empty() || symbolic.supernode_ptr.front() != 0 ||
            symbolic.supernode_ptr.back() != n || symbolic.row_ptr.front() != 0 ||
            symbolic.row_ptr.back() != static_cast<int>(symbolic.supernode_rows.size())) {
            throw std::invalid_argument("invalid CHOLMOD supernode structure");
        }
        for (std::size_t node = 0; node < node_count; ++node) {
            if (symbolic.supernode_ptr[node] >= symbolic.supernode_ptr[node + 1] ||
                symbolic.row_ptr[node] > symbolic.row_ptr[node + 1] ||
                symbolic.supernode_parent[node] >= static_cast<int>(node_count) ||
                symbolic.supernode_parent[node] == static_cast<int>(node)) {
                throw std::invalid_argument("invalid CHOLMOD supernode range or parent");
            }
        }
    }

    void resetNumericState()
    {
        nodes_.clear();
        factorization_order_.clear();
        base_owner_ptr_.clear();
        base_rows_.reset();
        base_cols_.reset();
        base_values_.reset();
        statistics_ = GpuLdltStatistics();
        n_ = 0;
        complete_ = false;
        diagnostic_.clear();
    }

    void prepareTree(const CholmodSymbolicResult& symbolic)
    {
        nodes_.resize(symbolic.supernode_parent.size());
        for (std::size_t node = 0; node < nodes_.size(); ++node) {
            nodes_[node].parent = symbolic.supernode_parent[node];
            if (nodes_[node].parent >= 0) {
                nodes_[static_cast<std::size_t>(nodes_[node].parent)].children.push_back(
                    static_cast<int>(node));
            }
        }
    }

    static std::uint64_t symmetricKey(int row, int col)
    {
        const std::uint32_t high = static_cast<std::uint32_t>(std::max(row, col));
        const std::uint32_t low = static_cast<std::uint32_t>(std::min(row, col));
        return (static_cast<std::uint64_t>(high) << 32) | low;
    }

    void prepareBaseEntries(
        int n,
        const std::vector<int>& col_ptr,
        const std::vector<int>& row_indices,
        const std::vector<float>& values,
        const CholmodSymbolicResult& symbolic)
    {
        std::unordered_map<std::uint64_t, SymmetricAccumulator> entries;
        entries.reserve(values.size());
        for (int col = 0; col < n; ++col) {
            for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
                const int row = row_indices[p];
                SymmetricAccumulator& accumulator = entries[symmetricKey(row, col)];
                if (row >= col) {
                    accumulator.lower += values[p];
                    ++accumulator.lower_count;
                } else {
                    accumulator.upper += values[p];
                    ++accumulator.upper_count;
                }
            }
        }

        std::vector<int> column_to_node(static_cast<std::size_t>(n), -1);
        for (std::size_t node = 0; node < nodes_.size(); ++node) {
            for (int col = symbolic.supernode_ptr[node];
                 col < symbolic.supernode_ptr[node + 1]; ++col) {
                column_to_node[col] = static_cast<int>(node);
            }
        }

        std::vector<std::vector<BaseEntry> > by_owner(nodes_.size());
        for (std::unordered_map<std::uint64_t, SymmetricAccumulator>::const_iterator
                 iterator = entries.begin(); iterator != entries.end(); ++iterator) {
            const int row = static_cast<int>(iterator->first >> 32);
            const int col = static_cast<int>(iterator->first & 0xffffffffu);
            const SymmetricAccumulator& accumulator = iterator->second;

            float value = 0.0f;
            if (row == col) {
                value = accumulator.lower + accumulator.upper;
            } else if (accumulator.lower_count != 0 && accumulator.upper_count != 0) {
                statistics_.maximum_input_asymmetry = std::max(
                    statistics_.maximum_input_asymmetry,
                    std::fabs(accumulator.lower - accumulator.upper));
                value = options_.symmetrize_input
                    ? 0.5f * (accumulator.lower + accumulator.upper)
                    : accumulator.lower;
            } else {
                value = accumulator.lower_count != 0
                    ? accumulator.lower : accumulator.upper;
            }
            if (value != 0.0f) {
                BaseEntry entry;
                entry.row = row;
                entry.col = col;
                entry.value = value;
                by_owner[static_cast<std::size_t>(column_to_node[col])].push_back(entry);
            }
        }

        base_owner_ptr_.assign(nodes_.size() + 1, 0);
        std::vector<int> host_rows;
        std::vector<int> host_cols;
        std::vector<float> host_values;
        host_rows.reserve(entries.size());
        host_cols.reserve(entries.size());
        host_values.reserve(entries.size());
        for (std::size_t node = 0; node < nodes_.size(); ++node) {
            std::vector<BaseEntry>& owned = by_owner[node];
            std::sort(owned.begin(), owned.end(),
                      [](const BaseEntry& lhs, const BaseEntry& rhs) {
                          return lhs.col < rhs.col ||
                                 (lhs.col == rhs.col && lhs.row < rhs.row);
                      });
            for (std::size_t i = 0; i < owned.size(); ++i) {
                host_rows.push_back(owned[i].row);
                host_cols.push_back(owned[i].col);
                host_values.push_back(owned[i].value);
            }
            base_owner_ptr_[node + 1] = static_cast<int>(host_rows.size());
        }

        base_rows_.allocate(host_rows.size());
        base_cols_.allocate(host_cols.size());
        base_values_.allocate(host_values.size());
        if (!host_rows.empty()) {
            checkCuda(cudaMemcpy(
                          base_rows_.get(), host_rows.data(), host_rows.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
                      "upload base entry rows");
            checkCuda(cudaMemcpy(
                          base_cols_.get(), host_cols.data(), host_cols.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
                      "upload base entry columns");
            checkCuda(cudaMemcpy(
                          base_values_.get(), host_values.data(),
                          host_values.size() * sizeof(float), cudaMemcpyHostToDevice),
                      "upload base entry values");
        }
    }

    static void appendUnique(
        std::vector<int>& destination,
        std::unordered_set<int>& seen,
        int value)
    {
        if (seen.insert(value).second) {
            destination.push_back(value);
        }
    }

    FrontWork assembleFront(int node_index, const CholmodSymbolicResult& symbolic)
    {
        FrontWork front;
        front.node = node_index;
        const std::size_t node = static_cast<std::size_t>(node_index);
        const int first_col = symbolic.supernode_ptr[node];
        const int end_col = symbolic.supernode_ptr[node + 1];

        std::unordered_set<int> seen;
        seen.reserve(static_cast<std::size_t>(
            symbolic.row_ptr[node + 1] - symbolic.row_ptr[node]) * 2 + 16);
        for (int col = first_col; col < end_col; ++col) {
            appendUnique(front.ids_host, seen, col);
        }

        NodeData& node_data = nodes_[node];
        for (std::size_t child_pos = 0; child_pos < node_data.children.size(); ++child_pos) {
            const NodeData& child = nodes_[static_cast<std::size_t>(
                node_data.children[child_pos])];
            for (int i = 0; i < child.delayed; ++i) {
                appendUnique(front.ids_host, seen, child.contribution_ids_host[i]);
            }
        }
        front.candidate_count = static_cast<int>(front.ids_host.size());

        for (int p = symbolic.row_ptr[node]; p < symbolic.row_ptr[node + 1]; ++p) {
            appendUnique(front.ids_host, seen, symbolic.supernode_rows[p]);
        }
        for (std::size_t child_pos = 0; child_pos < node_data.children.size(); ++child_pos) {
            const NodeData& child = nodes_[static_cast<std::size_t>(
                node_data.children[child_pos])];
            for (std::size_t i = static_cast<std::size_t>(child.delayed);
                 i < child.contribution_ids_host.size(); ++i) {
                appendUnique(front.ids_host, seen, child.contribution_ids_host[i]);
            }
        }
        front.order = static_cast<int>(front.ids_host.size());

        if (front.candidate_count == 1) {
            front.classification = GpuSupernodeClass::SingleColumn;
        } else if (front.candidate_count <= options_.batched_width_limit) {
            front.classification = GpuSupernodeClass::BatchedSmallMedium;
        } else {
            front.classification = GpuSupernodeClass::LargePanel;
        }

        front.values.allocate(
            static_cast<std::size_t>(front.order) * static_cast<std::size_t>(front.order));
        front.ids.allocate(front.ids_host.size());
        front.pivot_size.allocate(static_cast<std::size_t>(front.candidate_count));
        checkCuda(cudaMemset(
                      front.values.get(), 0,
                      static_cast<std::size_t>(front.order) * front.order * sizeof(float)),
                  "clear frontal matrix");
        checkCuda(cudaMemset(
                      front.pivot_size.get(), 0,
                      static_cast<std::size_t>(front.candidate_count) * sizeof(int)),
                  "clear pivot block sizes");
        checkCuda(cudaMemcpy(
                      front.ids.get(), front.ids_host.data(),
                      front.ids_host.size() * sizeof(int), cudaMemcpyHostToDevice),
                  "upload front IDs");

        std::vector<std::pair<int, int> > sorted;
        sorted.reserve(front.ids_host.size());
        for (int position = 0; position < front.order; ++position) {
            sorted.push_back(std::make_pair(front.ids_host[position], position));
        }
        std::sort(sorted.begin(), sorted.end());
        std::vector<int> sorted_ids(sorted.size());
        std::vector<int> sorted_positions(sorted.size());
        for (std::size_t i = 0; i < sorted.size(); ++i) {
            sorted_ids[i] = sorted[i].first;
            sorted_positions[i] = sorted[i].second;
        }
        front.sorted_ids.allocate(sorted_ids.size());
        front.sorted_positions.allocate(sorted_positions.size());
        checkCuda(cudaMemcpy(
                      front.sorted_ids.get(), sorted_ids.data(),
                      sorted_ids.size() * sizeof(int), cudaMemcpyHostToDevice),
                  "upload sorted front IDs");
        checkCuda(cudaMemcpy(
                      front.sorted_positions.get(), sorted_positions.data(),
                      sorted_positions.size() * sizeof(int), cudaMemcpyHostToDevice),
                  "upload sorted front positions");

        const int base_begin = base_owner_ptr_[node];
        const int base_end = base_owner_ptr_[node + 1];
        if (base_begin != base_end) {
            const int blocks = (base_end - base_begin + 255) / 256;
            scatterBaseEntriesKernel<<<blocks, 256>>>(
                front.values.get(), front.order,
                front.sorted_ids.get(), front.sorted_positions.get(), front.order,
                base_rows_.get(), base_cols_.get(), base_values_.get(),
                base_begin, base_end);
        }

        for (std::size_t child_pos = 0; child_pos < node_data.children.size(); ++child_pos) {
            const NodeData& child = nodes_[static_cast<std::size_t>(
                node_data.children[child_pos])];
            if (child.contribution_order == 0) {
                continue;
            }
            const int total = child.contribution_order * child.contribution_order;
            extendAddKernel<<<(total + 255) / 256, 256>>>(
                front.values.get(), front.order,
                front.sorted_ids.get(), front.sorted_positions.get(), front.order,
                child.contribution_values.get(), child.contribution_ids.get(),
                child.contribution_order);
        }
        checkCuda(cudaGetLastError(), "assemble frontal matrix on GPU");
        return front;
    }

    DeviceFrontDescriptor makeDescriptor(FrontWork& front) const
    {
        DeviceFrontDescriptor descriptor;
        descriptor.values = front.values.get();
        descriptor.ids = front.ids.get();
        descriptor.pivot_size = front.pivot_size.get();
        descriptor.order = front.order;
        descriptor.candidate_count = front.candidate_count;
        descriptor.gamma = options_.bunch_kaufman_gamma;
        descriptor.absolute_tolerance = options_.absolute_pivot_tolerance;
        descriptor.relative_tolerance = options_.relative_pivot_tolerance;
        descriptor.two_by_two_tolerance = options_.two_by_two_tolerance;
        descriptor.accepted = 0;
        descriptor.delayed = 0;
        descriptor.one_by_one = 0;
        descriptor.two_by_two = 0;
        return descriptor;
    }

    void factorBatch(
        std::vector<FrontWork>& fronts,
        const std::vector<int>& front_indices,
        bool single)
    {
        if (front_indices.empty()) {
            return;
        }
        std::vector<DeviceFrontDescriptor> host_descriptors;
        host_descriptors.reserve(front_indices.size());
        for (std::size_t i = 0; i < front_indices.size(); ++i) {
            host_descriptors.push_back(makeDescriptor(fronts[front_indices[i]]));
        }
        DeviceBuffer<DeviceFrontDescriptor> device_descriptors(host_descriptors.size());
        checkCuda(cudaMemcpy(
                      device_descriptors.get(), host_descriptors.data(),
                      host_descriptors.size() * sizeof(DeviceFrontDescriptor),
                      cudaMemcpyHostToDevice),
                  "upload batch descriptors");

        if (single) {
            factorSingleColumnKernel<<<static_cast<unsigned int>(host_descriptors.size()), 256>>>(
                device_descriptors.get());
        } else {
            factorSmallMediumKernel<<<static_cast<unsigned int>(host_descriptors.size()), 256>>>(
                device_descriptors.get());
        }
        checkCuda(cudaGetLastError(), "launch batched BK factorization");
        checkCuda(cudaMemcpy(
                      host_descriptors.data(), device_descriptors.get(),
                      host_descriptors.size() * sizeof(DeviceFrontDescriptor),
                      cudaMemcpyDeviceToHost),
                  "download batch results");

        for (std::size_t i = 0; i < front_indices.size(); ++i) {
            FrontWork& front = fronts[front_indices[i]];
            front.accepted = host_descriptors[i].accepted;
            front.delayed = host_descriptors[i].delayed;
            front.one_by_one = host_descriptors[i].one_by_one;
            front.two_by_two = host_descriptors[i].two_by_two;
        }
    }

    void flushLargePanel(
        FrontWork& front,
        DeviceBuffer<float>& work,
        int panel_begin,
        int step) const
    {
        const int panel_columns = step - panel_begin;
        const int remaining = front.order - step;
        if (panel_columns <= 0 || remaining <= 0) {
            return;
        }
        const float alpha = -1.0f;
        const float beta = 1.0f;
        checkCublas(cublasSgemm(
                        cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
                        remaining, remaining, panel_columns,
                        &alpha,
                        front.values.get() + step + panel_begin * front.order,
                        front.order,
                        work.get() + step,
                        front.order,
                        &beta,
                        front.values.get() + step + step * front.order,
                        front.order),
                    "flush large BK panel with SGEMM");
        const int total = remaining * remaining;
        mirrorLowerToUpperKernel<<<(total + 255) / 256, 256>>>(
            front.values.get(), front.order, step);
        checkCuda(cudaGetLastError(), "mirror SGEMM Schur update");
    }

    void factorLarge(FrontWork& front)
    {
        DeviceBuffer<float> work(
            static_cast<std::size_t>(front.order) * options_.panel_size);
        DeviceBuffer<DeviceLargePanelState> device_state(1);
        checkCuda(cudaMemset(
                      work.get(), 0,
                      static_cast<std::size_t>(front.order) * options_.panel_size *
                          sizeof(float)),
                  "clear large-panel work array");

        int step = 0;
        int active_end = front.candidate_count;
        int delay_first = 0;

        while (step < active_end) {
            const int panel_begin = step;
            factorLargePanelControlKernel<<<1, 256>>>(
                front.values.get(), work.get(), front.ids.get(),
                front.pivot_size.get(), front.order, panel_begin, active_end,
                delay_first, options_.panel_size, options_.bunch_kaufman_gamma,
                options_.absolute_pivot_tolerance,
                options_.relative_pivot_tolerance,
                options_.two_by_two_tolerance, device_state.get());
            checkCuda(cudaGetLastError(), "launch GPU-resident large BK panel");

            DeviceLargePanelState host_state;
            checkCuda(cudaMemcpy(
                          &host_state, device_state.get(), sizeof(host_state),
                          cudaMemcpyDeviceToHost),
                      "download large-panel summary");
            if (host_state.step < panel_begin ||
                host_state.active_end > active_end ||
                host_state.active_end < host_state.step ||
                host_state.panel_columns != host_state.step - panel_begin ||
                (host_state.delay_current != 0 &&
                 host_state.delay_current != 1)) {
                throw std::runtime_error("invalid GPU large-panel state");
            }

            step = host_state.step;
            active_end = host_state.active_end;
            front.one_by_one += host_state.one_by_one;
            front.two_by_two += host_state.two_by_two;
            delay_first = host_state.delay_current;
            if (host_state.panel_columns != 0) {
                flushLargePanel(front, work, panel_begin, step);
            }
        }

        front.accepted = step;
        front.delayed = front.candidate_count - step;
    }

    void finalizeFront(FrontWork& front)
    {
        const std::size_t node_index = static_cast<std::size_t>(front.node);
        NodeData& node = nodes_[node_index];
        node.order = front.order;
        node.accepted = front.accepted;
        node.delayed = front.delayed;

        front.ids_host.resize(static_cast<std::size_t>(front.order));
        checkCuda(cudaMemcpy(
                      front.ids_host.data(), front.ids.get(),
                      front.ids_host.size() * sizeof(int), cudaMemcpyDeviceToHost),
                  "download permuted front IDs");

        if (front.accepted != 0) {
            node.factor_values.allocate(
                static_cast<std::size_t>(front.order) * front.accepted);
            node.factor_ids.allocate(static_cast<std::size_t>(front.order));
            node.pivot_size.allocate(static_cast<std::size_t>(front.accepted));
            checkCuda(cudaMemcpy(
                          node.factor_values.get(), front.values.get(),
                          static_cast<std::size_t>(front.order) * front.accepted *
                              sizeof(float),
                          cudaMemcpyDeviceToDevice),
                      "save GPU supernode factor");
            checkCuda(cudaMemcpy(
                          node.factor_ids.get(), front.ids.get(),
                          static_cast<std::size_t>(front.order) * sizeof(int),
                          cudaMemcpyDeviceToDevice),
                      "save factor IDs");
            checkCuda(cudaMemcpy(
                          node.pivot_size.get(), front.pivot_size.get(),
                          static_cast<std::size_t>(front.accepted) * sizeof(int),
                          cudaMemcpyDeviceToDevice),
                      "save pivot block sizes");
        }

        node.contribution_order = front.order - front.accepted;
        node.contribution_ids_host.assign(
            front.ids_host.begin() + front.accepted, front.ids_host.end());
        if (node.contribution_order != 0) {
            node.contribution_values.allocate(
                static_cast<std::size_t>(node.contribution_order) *
                node.contribution_order);
            node.contribution_ids.allocate(
                static_cast<std::size_t>(node.contribution_order));
            const int total = node.contribution_order * node.contribution_order;
            extractContributionKernel<<<(total + 255) / 256, 256>>>(
                front.values.get(), front.order, front.accepted,
                node.contribution_values.get(), node.contribution_order);
            checkCuda(cudaMemcpy(
                          node.contribution_ids.get(),
                          front.ids.get() + front.accepted,
                          static_cast<std::size_t>(node.contribution_order) * sizeof(int),
                          cudaMemcpyDeviceToDevice),
                      "save contribution IDs");
        }
        checkCuda(cudaGetLastError(), "extract child contribution");

        statistics_.accepted_one_by_one_pivots +=
            static_cast<std::size_t>(front.one_by_one);
        statistics_.accepted_two_by_two_pivots +=
            static_cast<std::size_t>(front.two_by_two);
        statistics_.delayed_columns += static_cast<std::size_t>(front.delayed);
        if (node.parent < 0) {
            statistics_.unresolved_root_columns +=
                static_cast<std::size_t>(front.delayed);
        }
        factorization_order_.push_back(front.node);
    }

    void releaseConsumedChildContributions(const std::vector<int>& wave)
    {
        checkCuda(cudaDeviceSynchronize(), "finish wave contribution assembly");
        for (std::size_t i = 0; i < wave.size(); ++i) {
            NodeData& parent = nodes_[static_cast<std::size_t>(wave[i])];
            for (std::size_t child_pos = 0; child_pos < parent.children.size(); ++child_pos) {
                NodeData& child = nodes_[static_cast<std::size_t>(
                    parent.children[child_pos])];
                child.contribution_values.reset();
                child.contribution_ids.reset();
            }
        }
    }

    void runTreeWaves(const CholmodSymbolicResult& symbolic)
    {
        std::vector<int> pending_children(nodes_.size(), 0);
        std::vector<int> ready;
        for (std::size_t node = 0; node < nodes_.size(); ++node) {
            pending_children[node] = static_cast<int>(nodes_[node].children.size());
            if (pending_children[node] == 0) {
                ready.push_back(static_cast<int>(node));
            }
        }

        std::size_t processed = 0;
        while (!ready.empty()) {
            ++statistics_.tree_waves;
            std::vector<int> wave;
            wave.swap(ready);

            const std::size_t batch_limit =
                static_cast<std::size_t>(options_.max_batch_nodes);
            for (std::size_t batch_begin = 0;
                 batch_begin < wave.size(); batch_begin += batch_limit) {
                const std::size_t batch_end = std::min(
                    wave.size(), batch_begin + batch_limit);
                std::vector<int> batch_nodes(
                    wave.begin() + batch_begin, wave.begin() + batch_end);
                std::vector<FrontWork> fronts;
                fronts.reserve(batch_nodes.size());
                for (std::size_t i = 0; i < batch_nodes.size(); ++i) {
                    fronts.push_back(assembleFront(batch_nodes[i], symbolic));
                }
                releaseConsumedChildContributions(batch_nodes);

                std::vector<int> singles;
                std::vector<int> medium;
                std::vector<int> large;
                for (std::size_t i = 0; i < fronts.size(); ++i) {
                    switch (fronts[i].classification) {
                        case GpuSupernodeClass::SingleColumn:
                            singles.push_back(static_cast<int>(i));
                            ++statistics_.single_column_nodes;
                            break;
                        case GpuSupernodeClass::BatchedSmallMedium:
                            medium.push_back(static_cast<int>(i));
                            ++statistics_.batched_small_medium_nodes;
                            break;
                        case GpuSupernodeClass::LargePanel:
                            large.push_back(static_cast<int>(i));
                            ++statistics_.large_panel_nodes;
                            break;
                    }
                }

                factorBatch(fronts, singles, true);
                factorBatch(fronts, medium, false);
                for (std::size_t i = 0; i < large.size(); ++i) {
                    factorLarge(fronts[large[i]]);
                }

                for (std::size_t i = 0; i < fronts.size(); ++i) {
                    finalizeFront(fronts[i]);
                    ++processed;
                }
            }

            for (std::size_t i = 0; i < wave.size(); ++i) {
                const int parent = nodes_[static_cast<std::size_t>(wave[i])].parent;
                if (parent >= 0) {
                    int& count = pending_children[static_cast<std::size_t>(parent)];
                    --count;
                    if (count == 0) {
                        ready.push_back(parent);
                    }
                }
            }
        }

        if (processed != nodes_.size()) {
            throw std::runtime_error("the supernode parent relation contains a cycle");
        }
    }

    std::size_t acceptedVariableCount() const
    {
        std::size_t count = 0;
        for (std::size_t node = 0; node < nodes_.size(); ++node) {
            count += static_cast<std::size_t>(nodes_[node].accepted);
        }
        return count;
    }

    GpuLdltOptions options_;
    int n_;
    bool complete_;
    GpuLdltStatistics statistics_;
    std::string diagnostic_;
    cublasHandle_t cublas_;
    std::vector<NodeData> nodes_;
    std::vector<int> factorization_order_;
    std::vector<int> base_owner_ptr_;
    DeviceBuffer<int> base_rows_;
    DeviceBuffer<int> base_cols_;
    DeviceBuffer<float> base_values_;
};

GpuSupernodalLdltFactor::GpuSupernodalLdltFactor(const GpuLdltOptions& options)
    : impl_(new Impl(options))
{
}

GpuSupernodalLdltFactor::~GpuSupernodalLdltFactor() = default;

GpuSupernodalLdltFactor::GpuSupernodalLdltFactor(
    GpuSupernodalLdltFactor&& other) noexcept = default;

GpuSupernodalLdltFactor& GpuSupernodalLdltFactor::operator=(
    GpuSupernodalLdltFactor&& other) noexcept = default;

GpuLdltStatistics GpuSupernodalLdltFactor::factorize(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values,
    const CholmodSymbolicResult& symbolic)
{
    return impl_->factorize(n, col_ptr, row_indices, values, symbolic);
}

std::vector<float> GpuSupernodalLdltFactor::solve(
    const std::vector<float>& reordered_rhs) const
{
    return impl_->solve(reordered_rhs);
}

bool GpuSupernodalLdltFactor::complete() const
{
    return impl_->complete();
}

const GpuLdltStatistics& GpuSupernodalLdltFactor::statistics() const
{
    return impl_->statistics();
}

const std::string& GpuSupernodalLdltFactor::diagnostic() const
{
    return impl_->diagnostic();
}
