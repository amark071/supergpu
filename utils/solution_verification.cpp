#include "solution_verification.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

std::uint64_t symmetricKey(int row, int col)
{
    const std::uint32_t high =
        static_cast<std::uint32_t>(std::max(row, col));
    const std::uint32_t low =
        static_cast<std::uint32_t>(std::min(row, col));
    return (static_cast<std::uint64_t>(high) << 32) | low;
}

float symmetricValue(
    int row,
    int col,
    const SymmetricEntryAccumulator& accumulator,
    bool symmetrize_input)
{
    if (row == col) {
        return accumulator.lower + accumulator.upper;
    }
    if (accumulator.lower_count != 0 && accumulator.upper_count != 0) {
        return symmetrize_input
            ? 0.5f * (accumulator.lower + accumulator.upper)
            : accumulator.lower;
    }
    return accumulator.lower_count != 0
        ? accumulator.lower : accumulator.upper;
}

float infinityNorm(const std::vector<float>& values)
{
    float norm = 0.0f;
    for (std::size_t i = 0; i < values.size(); ++i) {
        norm = std::max(norm, std::fabs(values[i]));
    }
    return norm;
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

SymmetricEntryAccumulator::SymmetricEntryAccumulator()
    : lower(0.0f), upper(0.0f), lower_count(0), upper_count(0)
{
}

SymmetricEntries buildSymmetricEntries(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values)
{
    SymmetricEntries entries;
    entries.reserve(values.size());
    for (int col = 0; col < n; ++col) {
        for (int p = col_ptr[static_cast<std::size_t>(col)];
             p < col_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
            const int row = row_indices[static_cast<std::size_t>(p)];
            SymmetricEntryAccumulator& accumulator =
                entries[symmetricKey(row, col)];
            if (row >= col) {
                accumulator.lower += values[static_cast<std::size_t>(p)];
                ++accumulator.lower_count;
            } else {
                accumulator.upper += values[static_cast<std::size_t>(p)];
                ++accumulator.upper_count;
            }
        }
    }
    return entries;
}

std::vector<float> symmetricMatrixVectorProduct(
    int n,
    const SymmetricEntries& entries,
    bool symmetrize_input,
    const std::vector<float>& x,
    float* matrix_infinity_norm)
{
    std::vector<float> result(static_cast<std::size_t>(n), 0.0f);
    std::vector<float> absolute_row_sums(static_cast<std::size_t>(n), 0.0f);

    for (SymmetricEntries::const_iterator iterator = entries.begin();
         iterator != entries.end(); ++iterator) {
        const int row = static_cast<int>(iterator->first >> 32);
        const int col = static_cast<int>(iterator->first & 0xffffffffu);
        const float value = symmetricValue(
            row, col, iterator->second, symmetrize_input);
        result[static_cast<std::size_t>(row)] +=
            value * x[static_cast<std::size_t>(col)];
        absolute_row_sums[static_cast<std::size_t>(row)] += std::fabs(value);
        if (row != col) {
            result[static_cast<std::size_t>(col)] +=
                value * x[static_cast<std::size_t>(row)];
            absolute_row_sums[static_cast<std::size_t>(col)] += std::fabs(value);
        }
    }

    if (matrix_infinity_norm != 0) {
        *matrix_infinity_norm = *std::max_element(
            absolute_row_sums.begin(), absolute_row_sums.end());
    }
    return result;
}

ResidualVerification verifySolution(
    int n,
    const SymmetricEntries& entries,
    bool symmetrize_input,
    const std::vector<float>& rhs,
    const std::vector<float>& solution,
    const std::vector<float>& expected_solution)
{
    float matrix_norm = 0.0f;
    const std::vector<float> product = symmetricMatrixVectorProduct(
        n, entries, symmetrize_input, solution, &matrix_norm);

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

    ResidualVerification verification;
    verification.matrix_infinity_norm = matrix_norm;
    verification.rhs_infinity_norm = infinityNorm(rhs);
    verification.solution_infinity_norm = infinityNorm(solution);
    verification.absolute_residual = residual_norm;
    verification.relative_residual = safeRatio(
        residual_norm, verification.rhs_infinity_norm);
    verification.scaled_backward_error = safeRatio(
        residual_norm,
        verification.matrix_infinity_norm * verification.solution_infinity_norm +
            verification.rhs_infinity_norm);
    verification.relative_solution_error = safeRatio(
        solution_error, infinityNorm(expected_solution));
    return verification;
}
