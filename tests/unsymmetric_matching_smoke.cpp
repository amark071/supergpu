#include "unsymmetric_matching.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace {

double matchingWeight(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& rows,
    const std::vector<float>& values,
    const std::vector<int>& column_to_row)
{
    double result = 0.0;
    for (int col = 0; col < n; ++col) {
        const int matched_row = column_to_row[static_cast<std::size_t>(col)];
        double magnitude = 0.0;
        for (int p = col_ptr[static_cast<std::size_t>(col)];
             p < col_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
            if (rows[static_cast<std::size_t>(p)] == matched_row) {
                magnitude = std::max(
                    magnitude,
                    std::fabs(static_cast<double>(values[p])));
            }
        }
        if (magnitude == 0.0) {
            return -std::numeric_limits<double>::infinity();
        }
        result += std::log(magnitude);
    }
    return result;
}

double bruteForceBestWeight(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& rows,
    const std::vector<float>& values)
{
    std::vector<int> permutation(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        permutation[static_cast<std::size_t>(i)] = i;
    }
    double best = -std::numeric_limits<double>::infinity();
    do {
        best = std::max(
            best,
            matchingWeight(n, col_ptr, rows, values, permutation));
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return best;
}

} // namespace

int main()
{
    const int n = 4;
    const std::vector<int> col_ptr = {0, 4, 8, 12, 16};
    const std::vector<int> rows = {
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3};
    const std::vector<float> values = {
        1.0e-6f, 9.0f, 2.0f, 1.0f,
        8.0f, 1.0e-5f, 3.0f, 2.0f,
        1.0f, 2.0f, 1.0e-4f, 7.0f,
        2.0f, 3.0f, 6.0f, 1.0e-3f};

    const UnsymmetricMatching matching = computeDuffKosterMatching(
        n, col_ptr, rows, values);
    if (!matching.perfect || matching.cardinality != n) {
        std::cerr << "weighted matching did not find a perfect matching\n";
        return 1;
    }
    const double actual = matchingWeight(
        n, col_ptr, rows, values, matching.column_to_row);
    const double expected = bruteForceBestWeight(
        n, col_ptr, rows, values);
    if (std::fabs(actual - expected) > 1.0e-10) {
        std::cerr << "weighted matching is not maximum-product\n";
        return 2;
    }

    for (int col = 0; col < n; ++col) {
        const int row = matching.column_to_row[static_cast<std::size_t>(col)];
        float matched_value = 0.0f;
        for (int p = col_ptr[static_cast<std::size_t>(col)];
             p < col_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
            if (rows[static_cast<std::size_t>(p)] == row) {
                matched_value = values[static_cast<std::size_t>(p)];
                break;
            }
        }
        const float scaled = std::fabs(matched_value) *
            matching.row_scale[static_cast<std::size_t>(row)] *
            matching.column_scale[static_cast<std::size_t>(col)];
        if (std::fabs(scaled - 1.0f) > 1.0e-4f) {
            std::cerr << "matched diagonal scaling is not unit magnitude\n";
            return 3;
        }
    }

    unsigned int state = 17;
    for (int trial = 0; trial < 300; ++trial) {
        const int random_n = 2 + trial % 5;
        std::vector<int> random_col_ptr(
            static_cast<std::size_t>(random_n + 1), 0);
        std::vector<int> random_rows;
        std::vector<float> random_values;
        for (int col = 0; col < random_n; ++col) {
            for (int row = 0; row < random_n; ++row) {
                state = 1664525u * state + 1013904223u;
                if ((state >> 24) < 166u) {
                    random_rows.push_back(row);
                    const int magnitude = 1 + static_cast<int>((state >> 16) % 31u);
                    random_values.push_back(static_cast<float>(magnitude));
                }
            }
            random_col_ptr[static_cast<std::size_t>(col + 1)] =
                static_cast<int>(random_rows.size());
        }
        const double random_expected = bruteForceBestWeight(
            random_n, random_col_ptr, random_rows, random_values);
        const UnsymmetricMatching random_matching = computeDuffKosterMatching(
            random_n, random_col_ptr, random_rows, random_values);
        if (std::isfinite(random_expected)) {
            const double random_actual = matchingWeight(
                random_n, random_col_ptr, random_rows, random_values,
                random_matching.column_to_row);
            if (!random_matching.perfect ||
                std::fabs(random_actual - random_expected) > 1.0e-9) {
                std::cerr << "random weighted matching disagrees with brute force"
                          << " at trial " << trial << '\n';
                return 4;
            }
        }
    }

    std::cout << "Duff-Koster weighted matching smoke test passed\n";
    return 0;
}
