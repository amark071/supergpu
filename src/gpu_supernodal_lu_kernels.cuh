#ifndef SUPERNODAL_GPU_SRC_GPU_SUPERNODAL_LU_KERNELS_CUH
#define SUPERNODAL_GPU_SRC_GPU_SUPERNODAL_LU_KERNELS_CUH

#include <cuda_runtime.h>

#include <cmath>

namespace unsymmetric_lu_kernels {

__device__ inline bool usablePivot(float candidate, float column_max,
                                   float threshold, float zero_tolerance)
{
    return candidate > zero_tolerance &&
        candidate >= threshold * column_max;
}

__device__ inline int choosePivotRow(
    const float* matrix, int leading_dimension, int row_count,
    int candidate_count, int step, int col, float threshold,
    float zero_tolerance)
{
    float column_max = 0.0f;
    for (int row = step; row < row_count; ++row) {
        column_max = fmaxf(
            column_max, fabsf(matrix[row + col * leading_dimension]));
    }
    const float diagonal = fabsf(matrix[step + col * leading_dimension]);
    if (usablePivot(diagonal, column_max, threshold, zero_tolerance)) {
        return step;
    }
    int pivot = -1;
    float candidate_max = 0.0f;
    for (int row = step; row < candidate_count; ++row) {
        const float value = fabsf(matrix[row + col * leading_dimension]);
        if (value > candidate_max) {
            candidate_max = value;
            pivot = row;
        }
    }
    return usablePivot(candidate_max, column_max, threshold, zero_tolerance)
        ? pivot : -1;
}

__global__ void factorSmallFront(
    float* matrix, int leading_dimension, int row_count, int col_count,
    int candidate_count, int* row_ids, int* col_ids, float threshold,
    float zero_tolerance, int* accepted_out, int* delayed_events_out)
{
    __shared__ int step;
    __shared__ int active_end;
    __shared__ int pivot_row;
    __shared__ int replacement_col;
    __shared__ int delayed_events;
    if (threadIdx.x == 0) {
        step = 0;
        active_end = candidate_count;
        delayed_events = 0;
    }
    __syncthreads();

    while (step < active_end) {
        if (threadIdx.x == 0) {
            pivot_row = choosePivotRow(
                matrix, leading_dimension, row_count, candidate_count,
                step, step, threshold, zero_tolerance);
            replacement_col = step;
            if (pivot_row < 0) {
                replacement_col = -1;
                for (int col = active_end - 1; col > step; --col) {
                    const int row = choosePivotRow(
                        matrix, leading_dimension, row_count, candidate_count,
                        step, col, threshold, zero_tolerance);
                    if (row >= 0) {
                        replacement_col = col;
                        pivot_row = row;
                        break;
                    }
                }
                if (replacement_col < 0) {
                    delayed_events += active_end - step;
                    active_end = step;
                } else {
                    delayed_events += active_end - replacement_col;
                    active_end = replacement_col;
                }
            }
        }
        __syncthreads();
        if (step >= active_end) {
            break;
        }

        if (replacement_col != step) {
            for (int row = threadIdx.x; row < row_count; row += blockDim.x) {
                const int a = row + step * leading_dimension;
                const int b = row + replacement_col * leading_dimension;
                const float temporary = matrix[a];
                matrix[a] = matrix[b];
                matrix[b] = temporary;
            }
            if (threadIdx.x == 0) {
                const int temporary = col_ids[step];
                col_ids[step] = col_ids[replacement_col];
                col_ids[replacement_col] = temporary;
            }
        }
        __syncthreads();

        if (pivot_row != step) {
            for (int col = threadIdx.x; col < col_count; col += blockDim.x) {
                const int a = step + col * leading_dimension;
                const int b = pivot_row + col * leading_dimension;
                const float temporary = matrix[a];
                matrix[a] = matrix[b];
                matrix[b] = temporary;
            }
            if (threadIdx.x == 0) {
                const int temporary = row_ids[step];
                row_ids[step] = row_ids[pivot_row];
                row_ids[pivot_row] = temporary;
            }
        }
        __syncthreads();

        const float pivot = matrix[step + step * leading_dimension];
        for (int row = step + 1 + threadIdx.x;
             row < row_count; row += blockDim.x) {
            matrix[row + step * leading_dimension] /= pivot;
        }
        __syncthreads();

        const int update_rows = row_count - step - 1;
        const int update_cols = col_count - step - 1;
        for (int item = threadIdx.x; item < update_rows * update_cols;
             item += blockDim.x) {
            const int row = step + 1 + item % update_rows;
            const int col = step + 1 + item / update_rows;
            matrix[row + col * leading_dimension] -=
                matrix[row + step * leading_dimension] *
                matrix[step + col * leading_dimension];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            ++step;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *accepted_out = step;
        *delayed_events_out = delayed_events;
    }
}

__global__ void selectPivot(
    const float* matrix, int leading_dimension, int row_count,
    int candidate_count, int step, float threshold, float zero_tolerance,
    int* pivot_row)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *pivot_row = choosePivotRow(
            matrix, leading_dimension, row_count, candidate_count,
            step, step, threshold, zero_tolerance);
    }
}

__global__ void findReplacement(
    const float* matrix, int leading_dimension, int row_count,
    int candidate_count, int step, int active_end, float threshold,
    float zero_tolerance, int* replacement_col, int* new_active_end)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *replacement_col = -1;
        *new_active_end = step;
        for (int col = active_end - 1; col > step; --col) {
            if (choosePivotRow(
                    matrix, leading_dimension, row_count, candidate_count,
                    step, col, threshold, zero_tolerance) >= 0) {
                *replacement_col = col;
                *new_active_end = col;
                return;
            }
        }
    }
}

__global__ void swapRows(
    float* matrix, int leading_dimension, int col_count,
    int first, int second, int* row_ids)
{
    for (int col = blockIdx.x * blockDim.x + threadIdx.x;
         col < col_count; col += blockDim.x * gridDim.x) {
        const int a = first + col * leading_dimension;
        const int b = second + col * leading_dimension;
        const float temporary = matrix[a];
        matrix[a] = matrix[b];
        matrix[b] = temporary;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int temporary = row_ids[first];
        row_ids[first] = row_ids[second];
        row_ids[second] = temporary;
    }
}

__global__ void swapColumns(
    float* matrix, int leading_dimension, int row_count,
    int first, int second, int* col_ids)
{
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < row_count; row += blockDim.x * gridDim.x) {
        const int a = row + first * leading_dimension;
        const int b = row + second * leading_dimension;
        const float temporary = matrix[a];
        matrix[a] = matrix[b];
        matrix[b] = temporary;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int temporary = col_ids[first];
        col_ids[first] = col_ids[second];
        col_ids[second] = temporary;
    }
}

__global__ void dividePivotColumn(
    float* matrix, int leading_dimension, int row_count, int step)
{
    const float pivot = matrix[step + step * leading_dimension];
    for (int row = step + 1 + blockIdx.x * blockDim.x + threadIdx.x;
         row < row_count; row += blockDim.x * gridDim.x) {
        matrix[row + step * leading_dimension] /= pivot;
    }
}

__global__ void updatePanel(
    float* matrix, int leading_dimension, int row_count,
    int first_col, int end_col, int step)
{
    const int row = step + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    const int col = first_col + blockIdx.y * blockDim.y + threadIdx.y;
    if (row < row_count && col < end_col) {
        matrix[row + col * leading_dimension] -=
            matrix[row + step * leading_dimension] *
            matrix[step + col * leading_dimension];
    }
}

} // namespace unsymmetric_lu_kernels

#endif
