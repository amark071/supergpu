#ifndef SUPERNODAL_GPU_INCLUDE_UNSYMMETRIC_REFINEMENT_HPP
#define SUPERNODAL_GPU_INCLUDE_UNSYMMETRIC_REFINEMENT_HPP

#include "gpu_supernodal_lu.hpp"
#include "unsymmetric_ordering.hpp"

#include <vector>

struct UnsymmetricRefinementResult {
    std::vector<double> solution;
    int correction_steps = 0;
    double initial_residual_norm = 0.0;
    double final_residual_norm = 0.0;
};

UnsymmetricRefinementResult solveGeneralWithIterativeRefinement(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<double>& values,
    const std::vector<double>& rhs,
    const UnsymmetricOrdering& ordering,
    const GpuSupernodalLuFactor& factor,
    int maximum_corrections = 2);

#endif
