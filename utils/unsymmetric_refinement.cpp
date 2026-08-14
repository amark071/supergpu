#include "unsymmetric_refinement.hpp"

#include "unsymmetric_verification.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

std::vector<float> permuteRightHandSide(
    const std::vector<float>& rhs,
    const UnsymmetricOrdering& ordering)
{
    std::vector<float> result(rhs.size(), 0.0f);
    for (std::size_t new_row = 0; new_row < result.size(); ++new_row) {
        result[new_row] = rhs[static_cast<std::size_t>(
            ordering.row_perm[new_row])] * ordering.row_scale[new_row];
    }
    return result;
}

std::vector<float> restoreColumnOrder(
    const std::vector<float>& ordered_solution,
    const UnsymmetricOrdering& ordering)
{
    std::vector<float> result(ordered_solution.size(), 0.0f);
    for (std::size_t new_col = 0; new_col < result.size(); ++new_col) {
        result[static_cast<std::size_t>(ordering.perm[new_col])] =
            ordering.col_scale[new_col] * ordered_solution[new_col];
    }
    return result;
}

float infinityNorm(const std::vector<float>& values)
{
    float result = 0.0f;
    for (std::size_t i = 0; i < values.size(); ++i) {
        result = std::max(result, std::fabs(values[i]));
    }
    return result;
}

void applyCorrection(
    const std::vector<float>& correction,
    std::vector<float>& solution)
{
    for (std::size_t i = 0; i < solution.size(); ++i) {
        solution[i] += correction[i];
    }
}

} // namespace

UnsymmetricRefinementResult solveGeneralWithIterativeRefinement(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values,
    const std::vector<float>& rhs,
    const UnsymmetricOrdering& ordering,
    const GpuSupernodalLuFactor& factor,
    int maximum_corrections)
{
    if (maximum_corrections < 0 || maximum_corrections > 10) {
        throw std::invalid_argument("invalid iterative refinement limit");
    }
    UnsymmetricRefinementResult result;
    result.solution = restoreColumnOrder(
        factor.solve(permuteRightHandSide(rhs, ordering)), ordering);
    std::vector<float> best_solution = result.solution;
    float best_residual = std::numeric_limits<float>::infinity();
    for (int iteration = 0; iteration <= maximum_corrections; ++iteration) {
        const std::vector<float> residual = generalResidual(
            n, col_ptr, row_indices, values, rhs, result.solution);
        const float residual_norm = infinityNorm(residual);
        if (iteration == 0) {
            result.initial_residual_norm = residual_norm;
        }
        if (!std::isfinite(residual_norm) || residual_norm >= best_residual) {
            break;
        }
        best_residual = residual_norm;
        best_solution = result.solution;
        result.correction_steps = iteration;
        if (iteration == maximum_corrections) {
            break;
        }
        const std::vector<float> ordered_correction = factor.solve(
            permuteRightHandSide(residual, ordering));
        applyCorrection(
            restoreColumnOrder(ordered_correction, ordering), result.solution);
    }
    result.solution = std::move(best_solution);
    result.final_residual_norm = best_residual;
    return result;
}
