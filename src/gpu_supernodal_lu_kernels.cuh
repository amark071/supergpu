#ifndef SUPERNODAL_GPU_SRC_GPU_SUPERNODAL_LU_KERNELS_CUH
#define SUPERNODAL_GPU_SRC_GPU_SUPERNODAL_LU_KERNELS_CUH

#include <cuda_runtime.h>

#include <cmath>

namespace unsymmetric_lu_kernels {

using Scalar = float;

struct SmallFrontDescriptor {
    Scalar* matrix;
    int* row_ids;
    int* col_ids;
    int row_count;
    int col_count;
    int candidate_count;
    Scalar threshold;
    Scalar zero_tolerance;
    int accepted;
    int delayed;
};

struct LargePanelState {
    int step;
    int active_end;
    int updated_col_end;
};

__device__ inline bool usablePivot(Scalar candidate, Scalar column_max,
                                   Scalar threshold, Scalar zero_tolerance)
{
    return candidate > zero_tolerance &&
        candidate >= threshold * column_max;
}

__device__ inline int choosePivotRow(
    const Scalar* matrix, int leading_dimension, int row_count,
    int candidate_count, int step, int col, Scalar threshold,
    Scalar zero_tolerance)
{
    Scalar column_max = 0.0;
    for (int row = step; row < row_count; ++row) {
        column_max = fmax(
            column_max, fabs(matrix[row + col * leading_dimension]));
    }
    const Scalar diagonal = fabs(matrix[step + col * leading_dimension]);
    if (usablePivot(diagonal, column_max, threshold, zero_tolerance)) {
        return step;
    }
    int pivot = -1;
    Scalar candidate_max = 0.0;
    for (int row = step; row < candidate_count; ++row) {
        const Scalar value = fabs(matrix[row + col * leading_dimension]);
        if (value > candidate_max) {
            candidate_max = value;
            pivot = row;
        }
    }
    return usablePivot(candidate_max, column_max, threshold, zero_tolerance)
        ? pivot : -1;
}

__global__ void factorSmallFronts(SmallFrontDescriptor* descriptors)
{
    SmallFrontDescriptor& front = descriptors[blockIdx.x];
    Scalar* matrix = front.matrix;
    int* row_ids = front.row_ids;
    int* col_ids = front.col_ids;
    const int leading_dimension = front.row_count;
    const int row_count = front.row_count;
    const int col_count = front.col_count;
    const int candidate_count = front.candidate_count;
    const Scalar threshold = front.threshold;
    const Scalar zero_tolerance = front.zero_tolerance;
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
                const Scalar temporary = matrix[a];
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
                const Scalar temporary = matrix[a];
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

        const Scalar pivot = matrix[step + step * leading_dimension];
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
        front.accepted = step;
        front.delayed = delayed_events;
    }
}

__global__ void factorLargePanel(
    Scalar* matrix,
    int leading_dimension,
    int row_count,
    int col_count,
    int candidate_count,
    int* row_ids,
    int* col_ids,
    int panel_begin,
    int active_end_input,
    int panel_capacity,
    Scalar threshold,
    Scalar zero_tolerance,
    LargePanelState* output)
{
    __shared__ int step;
    __shared__ int active_end;
    __shared__ int updated_col_end;
    __shared__ int pivot_row;
    __shared__ int replacement_col;
    __shared__ int stop;
    if (threadIdx.x == 0) {
        step = panel_begin;
        active_end = active_end_input;
        updated_col_end = min(panel_begin + panel_capacity, active_end);
        stop = 0;
    }
    __syncthreads();

    while (step < updated_col_end) {
        if (threadIdx.x == 0) {
            pivot_row = choosePivotRow(
                matrix, leading_dimension, row_count, candidate_count,
                step, step, threshold, zero_tolerance);
            replacement_col = step;
            stop = 0;
            if (pivot_row < 0) {
                if (step != panel_begin) {
                    stop = 1;
                } else {
                    replacement_col = -1;
                    for (int col = active_end - 1; col > step; --col) {
                        const int row = choosePivotRow(
                            matrix, leading_dimension, row_count,
                            candidate_count, step, col,
                            threshold, zero_tolerance);
                        if (row >= 0) {
                            replacement_col = col;
                            pivot_row = row;
                            active_end = col;
                            updated_col_end = step + 1;
                            break;
                        }
                    }
                    if (replacement_col < 0) {
                        active_end = step;
                        updated_col_end = step;
                        stop = 1;
                    }
                }
            }
        }
        __syncthreads();
        if (stop) {
            break;
        }

        if (replacement_col != step) {
            for (int row = threadIdx.x; row < row_count; row += blockDim.x) {
                const int first = row + step * leading_dimension;
                const int second = row + replacement_col * leading_dimension;
                const Scalar temporary = matrix[first];
                matrix[first] = matrix[second];
                matrix[second] = temporary;
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
                const int first = step + col * leading_dimension;
                const int second = pivot_row + col * leading_dimension;
                const Scalar temporary = matrix[first];
                matrix[first] = matrix[second];
                matrix[second] = temporary;
            }
            if (threadIdx.x == 0) {
                const int temporary = row_ids[step];
                row_ids[step] = row_ids[pivot_row];
                row_ids[pivot_row] = temporary;
            }
        }
        __syncthreads();

        const Scalar pivot = matrix[step + step * leading_dimension];
        for (int row = step + 1 + threadIdx.x;
             row < row_count; row += blockDim.x) {
            matrix[row + step * leading_dimension] /= pivot;
        }
        __syncthreads();

        const int update_rows = row_count - step - 1;
        const int update_cols = updated_col_end - step - 1;
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
        output->step = step;
        output->active_end = active_end;
        output->updated_col_end = updated_col_end;
    }
}

} // namespace unsymmetric_lu_kernels

#endif
