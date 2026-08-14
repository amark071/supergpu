#include "unsymmetric_verification.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

float infinityNorm(const std::vector<float>& values)
{
    float result = 0.0f;
    for (std::size_t i = 0; i < values.size(); ++i) {
        result = std::max(result, std::fabs(values[i]));
    }
    return result;
}

float safeRatio(float numerator, float denominator)
{
    if (denominator > 0.0f) {
        return numerator / denominator;
    }
    return numerator == 0.0f
        ? 0.0f : std::numeric_limits<float>::infinity();
}

} // namespace

std::vector<float> generalMatrixVectorProduct(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values,
    const std::vector<float>& x,
    float* matrix_infinity_norm)
{
    if (n < 0 || col_ptr.size() != static_cast<std::size_t>(n + 1) ||
        row_indices.size() != values.size() ||
        x.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("invalid general CSC matrix-vector product");
    }
    std::vector<float> result(static_cast<std::size_t>(n), 0.0f);
    std::vector<float> row_sums(static_cast<std::size_t>(n), 0.0f);
    for (int col = 0; col < n; ++col) {
        for (int p = col_ptr[static_cast<std::size_t>(col)];
             p < col_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
            const int row = row_indices[static_cast<std::size_t>(p)];
            const float value = values[static_cast<std::size_t>(p)];
            result[static_cast<std::size_t>(row)] +=
                value * x[static_cast<std::size_t>(col)];
            row_sums[static_cast<std::size_t>(row)] += std::fabs(value);
        }
    }
    if (matrix_infinity_norm != 0) {
        *matrix_infinity_norm = row_sums.empty()
            ? 0.0f : *std::max_element(row_sums.begin(), row_sums.end());
    }
    return result;
}

ResidualVerification verifyGeneralSolution(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values,
    const std::vector<float>& rhs,
    const std::vector<float>& solution,
    const std::vector<float>& expected_solution)
{
    float matrix_norm = 0.0f;
    const std::vector<float> product = generalMatrixVectorProduct(
        n, col_ptr, row_indices, values, solution, &matrix_norm);
    float residual_norm = 0.0f;
    float solution_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        residual_norm = std::max(
            residual_norm,
            std::fabs(product[static_cast<std::size_t>(i)] -
                      rhs[static_cast<std::size_t>(i)]));
        solution_error = std::max(
            solution_error,
            std::fabs(solution[static_cast<std::size_t>(i)] -
                      expected_solution[static_cast<std::size_t>(i)]));
    }
    ResidualVerification result;
    result.matrix_infinity_norm = matrix_norm;
    result.rhs_infinity_norm = infinityNorm(rhs);
    result.solution_infinity_norm = infinityNorm(solution);
    result.absolute_residual = residual_norm;
    result.relative_residual = safeRatio(residual_norm, result.rhs_infinity_norm);
    result.scaled_backward_error = safeRatio(
        residual_norm,
        result.matrix_infinity_norm * result.solution_infinity_norm +
            result.rhs_infinity_norm);
    result.relative_solution_error = safeRatio(
        solution_error, infinityNorm(expected_solution));
    return result;
}
