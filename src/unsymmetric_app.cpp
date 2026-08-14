#include "unsymmetric_app.hpp"

#include "gpu_supernodal_lu.hpp"
#include "io_func.hpp"
#include "unsymmetric_ordering.hpp"
#include "unsymmetric_symbolic.hpp"
#include "unsymmetric_verification.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct InputMatrix {
    int n = 0;
    int declared_nonzeros = 0;
    std::vector<int> row_indices;
    std::vector<int> col_ptr;
    std::vector<float> values;
};

bool requestedOrderedExport(int argc, char** argv)
{
    for (int argument = 1; argument < argc; ++argument) {
        if (std::string(argv[argument]) == "--export-ordered-matrix") {
            return true;
        }
    }
    return false;
}

InputMatrix loadMatrix(const std::string& filename)
{
    InputMatrix matrix;
    readCSCMatrix(
        filename, matrix.n, matrix.declared_nonzeros,
        matrix.row_indices, matrix.col_ptr, matrix.values);
    return matrix;
}

std::vector<float> permuteRightHandSide(
    const std::vector<float>& rhs,
    const UnsymmetricOrdering& ordering)
{
    std::vector<float> result(rhs.size(), 0.0f);
    for (std::size_t new_row = 0; new_row < result.size(); ++new_row) {
        result[new_row] = rhs[static_cast<std::size_t>(
            ordering.row_perm[new_row])];
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
            ordered_solution[new_col];
    }
    return result;
}

void printSymbolicReport(
    const InputMatrix& input,
    const UnsymmetricOrdering& ordering,
    const UnsymmetricPermutedCsc& ordered,
    const UnsymmetricSymbolicResult& symbolic,
    float ordering_ms,
    float permutation_ms,
    float symbolic_ms)
{
    std::cout << "\n=== General/unsymmetric sparse LU ===\n";
    std::cout << "Matrix = " << input.n << 'x' << input.n
              << ", nonzero = " << ordered.values.size() << '\n';
    std::cout << "Column ordering = COLAMD (CHOLMOD not used)\n";
    std::cout << "Structural matching rank = " << ordering.structural_rank
              << " / " << input.n << '\n';
    std::cout << "COLAMD + structural matching time (ms) = "
              << ordering_ms << '\n';
    std::cout << "Unsymmetric row/column permutation time (ms) = "
              << permutation_ms << '\n';
    std::cout << "In-project symbolic analysis time (ms) = "
              << symbolic_ms << '\n';
    std::cout << "Structural envelope off-diagonal entries = "
              << symbolic.envelope_off_diagonal_nonzeros << '\n';
    std::cout << "Symbolic factor nonzeros = "
              << symbolic.symbolic_factor_nonzeros << '\n';
    std::cout << "Unsymmetric supernodes = "
              << symbolic.supernode_parent.size() << '\n';
}

void printFactorReport(
    const GpuLuStatistics& statistics,
    const GpuSupernodalLuFactor& factor)
{
    std::cout << "GPU LU single-column fronts = "
              << statistics.single_column_nodes << '\n';
    std::cout << "GPU LU 2--64 column fronts = "
              << statistics.small_medium_nodes << '\n';
    std::cout << "GPU LU >64 column fronts = "
              << statistics.large_panel_nodes << '\n';
    std::cout << "Accepted unsymmetric pivots = "
              << statistics.accepted_pivots << '\n';
    std::cout << "Delayed unsymmetric columns = "
              << statistics.delayed_columns << '\n';
    std::cout << "Unresolved root columns = "
              << statistics.unresolved_root_columns << '\n';
    std::cout << "Unsymmetric tree waves = " << statistics.tree_waves << '\n';
    std::cout << "Concurrent CUDA fronts = "
              << statistics.concurrent_fronts << '\n';
    std::cout << "Unsymmetric front assembly time (ms) = "
              << statistics.front_assembly_milliseconds << '\n';
    std::cout << "Small/medium LU GPU work sum (ms) = "
              << statistics.small_medium_factorization_milliseconds << '\n';
    std::cout << "Large panel LU GPU work sum (ms) = "
              << statistics.large_panel_factorization_milliseconds << '\n';
    std::cout << "Unsymmetric GPU factorization wall time (ms) = "
              << statistics.factorization_milliseconds << '\n';
    std::cout << "GPU general LU status = " << factor.diagnostic() << '\n';
}

bool printVerification(const ResidualVerification& result, float solve_ms)
{
    const float tolerance = std::sqrt(std::numeric_limits<float>::epsilon());
    const bool passed = std::isfinite(result.scaled_backward_error) &&
        result.scaled_backward_error <= tolerance;
    std::cout << std::scientific << std::setprecision(8);
    std::cout << "Unsymmetric solve verification x_true = all ones\n";
    std::cout << "Unsymmetric solve time (ms) = " << solve_ms << '\n';
    std::cout << "Absolute residual ||A*x-b||_inf = "
              << result.absolute_residual << '\n';
    std::cout << "Scaled backward error = "
              << result.scaled_backward_error << '\n';
    std::cout << "Relative solution error = "
              << result.relative_solution_error << '\n';
    std::cout << "Unsymmetric residual verification = "
              << (passed ? "PASS" : "WARNING") << '\n';
    return passed;
}

} // namespace

int runUnsymmetricGpuApplication(
    int argc,
    char** argv,
    const UnsymmetricAppFiles& files)
{
    try {
        const InputMatrix input = loadMatrix(files.input_filename);
        const std::chrono::steady_clock::time_point ordering_begin =
            std::chrono::steady_clock::now();
        const UnsymmetricOrdering ordering = computeColamdOrdering(
            input.n, input.col_ptr, input.row_indices);
        const float ordering_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - ordering_begin).count();

        const std::chrono::steady_clock::time_point permutation_begin =
            std::chrono::steady_clock::now();
        const UnsymmetricPermutedCsc ordered = applyUnsymmetricPermutationCsc(
            input.n, input.col_ptr, input.row_indices, input.values, ordering);
        const float permutation_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - permutation_begin).count();
        if (requestedOrderedExport(argc, argv)) {
            writeCSCMatrix(
                files.ordered_output_filename, input.n, ordered.row_indices,
                ordered.col_ptr, ordered.values);
        }

        const std::chrono::steady_clock::time_point symbolic_begin =
            std::chrono::steady_clock::now();
        const UnsymmetricSymbolicResult symbolic = analyzeUnsymmetricSupernodes(
            input.n, ordered.col_ptr, ordered.row_indices);
        const float symbolic_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - symbolic_begin).count();
        printSymbolicReport(
            input, ordering, ordered, symbolic,
            ordering_ms, permutation_ms, symbolic_ms);

        GpuLuOptions gpu_options;
        GpuSupernodalLuFactor factor(gpu_options);
        const GpuLuStatistics statistics = factor.factorize(
            input.n, ordered.col_ptr, ordered.row_indices,
            ordered.values, symbolic);
        printFactorReport(statistics, factor);
        if (!factor.complete()) {
            return 4;
        }

        const std::vector<float> expected(
            static_cast<std::size_t>(input.n), 1.0f);
        const std::vector<float> rhs = generalMatrixVectorProduct(
            input.n, input.col_ptr, input.row_indices,
            input.values, expected, 0);
        const std::vector<float> ordered_rhs =
            permuteRightHandSide(rhs, ordering);
        const std::chrono::steady_clock::time_point solve_begin =
            std::chrono::steady_clock::now();
        const std::vector<float> ordered_solution = factor.solve(ordered_rhs);
        const std::vector<float> solution =
            restoreColumnOrder(ordered_solution, ordering);
        const float solve_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - solve_begin).count();
        const ResidualVerification verification = verifyGeneralSolution(
            input.n, input.col_ptr, input.row_indices, input.values,
            rhs, solution, expected);
        return printVerification(verification, solve_ms) ? 0 : 5;
    } catch (const std::exception& error) {
        std::cerr << "Unsymmetric GPU LU failed: " << error.what() << '\n';
        return 6;
    }
}
