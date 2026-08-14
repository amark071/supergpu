#include "unsymmetric_matching.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

const int kNoIndex = -1;
const double kZeroTolerance = 1.0e-12;
const double kInfinity = std::numeric_limits<double>::infinity();
typedef std::pair<double, int> QueueEntry;
typedef std::priority_queue<
    QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry> > SearchQueue;

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
        throw std::invalid_argument("invalid zero-based CSC matrix");
    }
    for (int col = 0; col < n; ++col) {
        if (col_ptr[static_cast<std::size_t>(col)] >
            col_ptr[static_cast<std::size_t>(col + 1)]) {
            throw std::invalid_argument("CSC column pointers are not monotonic");
        }
    }
    for (std::size_t p = 0; p < row_indices.size(); ++p) {
        if (row_indices[p] < 0 || row_indices[p] >= n ||
            !std::isfinite(values[p])) {
            throw std::invalid_argument("invalid CSC entry");
        }
    }
}

class DuffKosterMatcher {
public:
    DuffKosterMatcher(
        int n,
        const std::vector<int>& col_ptr,
        const std::vector<int>& row_indices,
        const std::vector<float>& values)
        : n_(n),
          col_ptr_(col_ptr),
          row_indices_(row_indices),
          values_(values),
          cost_(values.size(), kInfinity),
          column_max_log_(static_cast<std::size_t>(n), -kInfinity),
          row_dual_(static_cast<std::size_t>(n), 0.0),
          column_dual_(static_cast<std::size_t>(n), 0.0),
          row_match_(static_cast<std::size_t>(n), kNoIndex),
          column_match_(static_cast<std::size_t>(n), kNoIndex),
          match_pointer_(static_cast<std::size_t>(n), kNoIndex),
          distance_(static_cast<std::size_t>(n), kInfinity),
          previous_column_(static_cast<std::size_t>(n), kNoIndex),
          previous_pointer_(static_cast<std::size_t>(n), kNoIndex),
          settled_(static_cast<std::size_t>(n), 0),
          queued_(static_cast<std::size_t>(n), 0),
          update_mark_(static_cast<std::size_t>(n), 0)
    {
        touched_.reserve(static_cast<std::size_t>(n));
        update_columns_.reserve(static_cast<std::size_t>(n));
    }

    UnsymmetricMatching run()
    {
        precomputeCosts();
        initialExtremeMatching();
        for (int col = 0; col < n_; ++col) {
            if (column_match_[static_cast<std::size_t>(col)] < 0) {
                findShortestAugmentingPath(col);
            }
        }
        completeStructuralMatching();
        return makeResult();
    }

private:
    double reducedCost(int row, int col, int pointer) const
    {
        return cost_[static_cast<std::size_t>(pointer)] -
            row_dual_[static_cast<std::size_t>(row)] -
            column_dual_[static_cast<std::size_t>(col)];
    }

    void precomputeCosts()
    {
        for (int col = 0; col < n_; ++col) {
            double maximum = 0.0;
            for (int p = col_ptr_[static_cast<std::size_t>(col)];
                 p < col_ptr_[static_cast<std::size_t>(col + 1)]; ++p) {
                maximum = std::max(
                    maximum,
                    std::fabs(static_cast<double>(values_[p])));
            }
            if (maximum > 0.0) {
                column_max_log_[static_cast<std::size_t>(col)] =
                    std::log(maximum);
            }
        }
        for (int col = 0; col < n_; ++col) {
            const double maximum_log =
                column_max_log_[static_cast<std::size_t>(col)];
            for (int p = col_ptr_[static_cast<std::size_t>(col)];
                 p < col_ptr_[static_cast<std::size_t>(col + 1)]; ++p) {
                const double magnitude =
                    std::fabs(static_cast<double>(values_[p]));
                if (magnitude > 0.0 && std::isfinite(maximum_log)) {
                    double entry_cost = maximum_log - std::log(magnitude);
                    if (entry_cost < 0.0 && entry_cost > -1.0e-13) {
                        entry_cost = 0.0;
                    }
                    cost_[static_cast<std::size_t>(p)] = entry_cost;
                }
            }
        }
    }

    void initialExtremeMatching()
    {
        std::fill(row_dual_.begin(), row_dual_.end(), kInfinity);
        for (int col = 0; col < n_; ++col) {
            for (int p = col_ptr_[static_cast<std::size_t>(col)];
                 p < col_ptr_[static_cast<std::size_t>(col + 1)]; ++p) {
                const int row = row_indices_[static_cast<std::size_t>(p)];
                row_dual_[static_cast<std::size_t>(row)] = std::min(
                    row_dual_[static_cast<std::size_t>(row)],
                    cost_[static_cast<std::size_t>(p)]);
            }
        }
        for (int row = 0; row < n_; ++row) {
            if (!std::isfinite(row_dual_[static_cast<std::size_t>(row)])) {
                row_dual_[static_cast<std::size_t>(row)] = 0.0;
            }
        }
        initializeColumnDuals();
        greedilyMatchZeroCostEdges();
        augmentLengthTwoZeroCostPaths();
    }

    void initializeColumnDuals()
    {
        for (int col = 0; col < n_; ++col) {
            double minimum = kInfinity;
            for (int p = col_ptr_[static_cast<std::size_t>(col)];
                 p < col_ptr_[static_cast<std::size_t>(col + 1)]; ++p) {
                const int row = row_indices_[static_cast<std::size_t>(p)];
                minimum = std::min(
                    minimum,
                    cost_[static_cast<std::size_t>(p)] -
                        row_dual_[static_cast<std::size_t>(row)]);
            }
            column_dual_[static_cast<std::size_t>(col)] =
                std::isfinite(minimum) ? minimum : 0.0;
        }
    }

    void greedilyMatchZeroCostEdges()
    {
        for (int col = 0; col < n_; ++col) {
            for (int p = col_ptr_[static_cast<std::size_t>(col)];
                 p < col_ptr_[static_cast<std::size_t>(col + 1)]; ++p) {
                const int row = row_indices_[static_cast<std::size_t>(p)];
                if (row_match_[static_cast<std::size_t>(row)] < 0 &&
                    std::fabs(reducedCost(row, col, p)) <= kZeroTolerance) {
                    assign(row, col, p);
                    break;
                }
            }
        }
    }

    void augmentLengthTwoZeroCostPaths()
    {
        for (int col = 0; col < n_; ++col) {
            if (column_match_[static_cast<std::size_t>(col)] >= 0) {
                continue;
            }
            for (int p = col_ptr_[static_cast<std::size_t>(col)];
                 p < col_ptr_[static_cast<std::size_t>(col + 1)]; ++p) {
                const int row = row_indices_[static_cast<std::size_t>(p)];
                if (std::fabs(reducedCost(row, col, p)) > kZeroTolerance) {
                    continue;
                }
                if (row_match_[static_cast<std::size_t>(row)] < 0) {
                    assign(row, col, p);
                    break;
                }
                const int displaced_col =
                    row_match_[static_cast<std::size_t>(row)];
                bool augmented = false;
                for (int p2 = col_ptr_[static_cast<std::size_t>(displaced_col)];
                     p2 < col_ptr_[static_cast<std::size_t>(displaced_col + 1)];
                     ++p2) {
                    const int free_row =
                        row_indices_[static_cast<std::size_t>(p2)];
                    if (row_match_[static_cast<std::size_t>(free_row)] < 0 &&
                        std::fabs(reducedCost(
                            free_row, displaced_col, p2)) <= kZeroTolerance) {
                        assign(row, col, p);
                        assign(free_row, displaced_col, p2);
                        augmented = true;
                        break;
                    }
                }
                if (augmented) {
                    break;
                }
            }
        }
    }

    void assign(int row, int col, int pointer)
    {
        row_match_[static_cast<std::size_t>(row)] = col;
        column_match_[static_cast<std::size_t>(col)] = row;
        match_pointer_[static_cast<std::size_t>(col)] = pointer;
    }

    void resetSearch()
    {
        for (std::size_t p = 0; p < touched_.size(); ++p) {
            const int row = touched_[p];
            distance_[static_cast<std::size_t>(row)] = kInfinity;
            previous_column_[static_cast<std::size_t>(row)] = kNoIndex;
            previous_pointer_[static_cast<std::size_t>(row)] = kNoIndex;
            settled_[static_cast<std::size_t>(row)] = 0;
            queued_[static_cast<std::size_t>(row)] = 0;
        }
        touched_.clear();
    }

    bool findShortestAugmentingPath(int initial_col)
    {
        SearchQueue queue;
        resetSearch();
        double settled_distance = 0.0;
        double augmenting_distance = kInfinity;
        int augmenting_row = kNoIndex;
        int augmenting_pointer = kNoIndex;
        int col = initial_col;

        while (true) {
            scanColumn(
                col, settled_distance, augmenting_distance,
                augmenting_row, augmenting_pointer, queue);
            int next_row = kNoIndex;
            while (!queue.empty()) {
                const QueueEntry entry = queue.top();
                queue.pop();
                if (!settled_[static_cast<std::size_t>(entry.second)] &&
                    entry.first == distance_[static_cast<std::size_t>(entry.second)]) {
                    next_row = entry.second;
                    break;
                }
            }
            if (next_row < 0) {
                break;
            }
            settled_distance = distance_[static_cast<std::size_t>(next_row)];
            if (augmenting_distance <= settled_distance) {
                break;
            }
            settled_[static_cast<std::size_t>(next_row)] = 1;
            queued_[static_cast<std::size_t>(next_row)] = 0;
            col = row_match_[static_cast<std::size_t>(next_row)];
        }
        if (augmenting_row < 0) {
            return false;
        }
        updateDualsAndAugment(
            augmenting_row, augmenting_pointer, augmenting_distance);
        return true;
    }

    void scanColumn(
        int col,
        double settled_distance,
        double& augmenting_distance,
        int& augmenting_row,
        int& augmenting_pointer,
        SearchQueue& queue)
    {
        for (int p = col_ptr_[static_cast<std::size_t>(col)];
             p < col_ptr_[static_cast<std::size_t>(col + 1)]; ++p) {
            const int row = row_indices_[static_cast<std::size_t>(p)];
            if (settled_[static_cast<std::size_t>(row)]) {
                continue;
            }
            double reduced = reducedCost(row, col, p);
            if (reduced < 0.0 && reduced > -1.0e-10) {
                reduced = 0.0;
            }
            const double candidate = settled_distance + reduced;
            if (!(candidate < augmenting_distance)) {
                continue;
            }
            if (row_match_[static_cast<std::size_t>(row)] < 0) {
                augmenting_distance = candidate;
                augmenting_row = row;
                augmenting_pointer = p;
                previous_column_[static_cast<std::size_t>(row)] = col;
                previous_pointer_[static_cast<std::size_t>(row)] = p;
                touchRow(row);
            } else if (candidate < distance_[static_cast<std::size_t>(row)]) {
                if (!std::isfinite(distance_[static_cast<std::size_t>(row)])) {
                    touched_.push_back(row);
                }
                distance_[static_cast<std::size_t>(row)] = candidate;
                previous_column_[static_cast<std::size_t>(row)] = col;
                previous_pointer_[static_cast<std::size_t>(row)] = p;
                queued_[static_cast<std::size_t>(row)] = 1;
                queue.push(std::make_pair(candidate, row));
            }
        }
    }

    void touchRow(int row)
    {
        if (!queued_[static_cast<std::size_t>(row)]) {
            queued_[static_cast<std::size_t>(row)] = 1;
            touched_.push_back(row);
        }
    }

    void markUpdateColumn(int col)
    {
        if (col >= 0 && update_mark_[static_cast<std::size_t>(col)] != stamp_) {
            update_mark_[static_cast<std::size_t>(col)] = stamp_;
            update_columns_.push_back(col);
        }
    }

    void updateDualsAndAugment(
        int augmenting_row,
        int augmenting_pointer,
        double augmenting_distance)
    {
        update_columns_.clear();
        ++stamp_;
        if (stamp_ == std::numeric_limits<int>::max()) {
            std::fill(update_mark_.begin(), update_mark_.end(), 0);
            stamp_ = 1;
        }
        for (std::size_t p = 0; p < touched_.size(); ++p) {
            const int row = touched_[p];
            if (settled_[static_cast<std::size_t>(row)]) {
                row_dual_[static_cast<std::size_t>(row)] +=
                    distance_[static_cast<std::size_t>(row)] -
                    augmenting_distance;
                markUpdateColumn(row_match_[static_cast<std::size_t>(row)]);
            }
        }

        int row = augmenting_row;
        int pointer = augmenting_pointer;
        while (row >= 0) {
            const int col = previous_column_[static_cast<std::size_t>(row)];
            const int displaced_row =
                column_match_[static_cast<std::size_t>(col)];
            assign(row, col, pointer);
            markUpdateColumn(col);
            if (displaced_row < 0) {
                break;
            }
            pointer = previous_pointer_[static_cast<std::size_t>(displaced_row)];
            row = displaced_row;
        }
        for (std::size_t p = 0; p < update_columns_.size(); ++p) {
            const int col = update_columns_[p];
            const int matched_row =
                column_match_[static_cast<std::size_t>(col)];
            const int matched_pointer =
                match_pointer_[static_cast<std::size_t>(col)];
            if (matched_row >= 0 && matched_pointer >= 0) {
                column_dual_[static_cast<std::size_t>(col)] =
                    cost_[static_cast<std::size_t>(matched_pointer)] -
                    row_dual_[static_cast<std::size_t>(matched_row)];
            }
        }
    }

    bool augmentStructurally(
        int col,
        std::vector<int>& visited_rows,
        int stamp)
    {
        for (int p = col_ptr_[static_cast<std::size_t>(col)];
             p < col_ptr_[static_cast<std::size_t>(col + 1)]; ++p) {
            if (values_[static_cast<std::size_t>(p)] == 0.0f) {
                continue;
            }
            const int row = row_indices_[static_cast<std::size_t>(p)];
            if (visited_rows[static_cast<std::size_t>(row)] == stamp) {
                continue;
            }
            visited_rows[static_cast<std::size_t>(row)] = stamp;
            const int displaced_col =
                row_match_[static_cast<std::size_t>(row)];
            if (displaced_col < 0 ||
                augmentStructurally(displaced_col, visited_rows, stamp)) {
                assign(row, col, p);
                return true;
            }
        }
        return false;
    }

    void completeStructuralMatching()
    {
        std::vector<int> visited_rows(static_cast<std::size_t>(n_), 0);
        int search_stamp = 0;
        bool augmented = true;
        while (augmented) {
            augmented = false;
            for (int col = 0; col < n_; ++col) {
                if (column_match_[static_cast<std::size_t>(col)] >= 0) {
                    continue;
                }
                ++search_stamp;
                if (augmentStructurally(col, visited_rows, search_stamp)) {
                    augmented = true;
                }
            }
        }
    }

    UnsymmetricMatching makeResult()
    {
        UnsymmetricMatching result;
        result.row_to_column = row_match_;
        result.column_to_row = column_match_;
        result.row_scale.resize(static_cast<std::size_t>(n_), 1.0f);
        result.column_scale.resize(static_cast<std::size_t>(n_), 1.0f);
        for (int col = 0; col < n_; ++col) {
            if (column_match_[static_cast<std::size_t>(col)] >= 0) {
                ++result.cardinality;
            }
        }
        result.perfect = result.cardinality == n_;
        for (int row = 0; row < n_; ++row) {
            result.row_scale[static_cast<std::size_t>(row)] =
                safeExp(row_dual_[static_cast<std::size_t>(row)]);
        }
        for (int col = 0; col < n_; ++col) {
            const double maximum_log =
                column_max_log_[static_cast<std::size_t>(col)];
            if (std::isfinite(maximum_log)) {
                result.column_scale[static_cast<std::size_t>(col)] = safeExp(
                    column_dual_[static_cast<std::size_t>(col)] - maximum_log);
            }
        }
        return result;
    }

    static float safeExp(double exponent)
    {
        const double maximum = std::log(
            static_cast<double>(std::numeric_limits<float>::max()));
        const double minimum = std::log(
            static_cast<double>(std::numeric_limits<float>::min()));
        return static_cast<float>(std::exp(
            std::max(minimum, std::min(maximum, exponent))));
    }

    int n_;
    const std::vector<int>& col_ptr_;
    const std::vector<int>& row_indices_;
    const std::vector<float>& values_;
    std::vector<double> cost_;
    std::vector<double> column_max_log_;
    std::vector<double> row_dual_;
    std::vector<double> column_dual_;
    std::vector<int> row_match_;
    std::vector<int> column_match_;
    std::vector<int> match_pointer_;
    std::vector<double> distance_;
    std::vector<int> previous_column_;
    std::vector<int> previous_pointer_;
    std::vector<unsigned char> settled_;
    std::vector<unsigned char> queued_;
    std::vector<int> touched_;
    std::vector<int> update_columns_;
    std::vector<int> update_mark_;
    int stamp_ = 0;
};

} // namespace

UnsymmetricMatching computeDuffKosterMatching(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    const std::vector<float>& values)
{
    validateCsc(n, col_ptr, row_indices, values);
    return DuffKosterMatcher(n, col_ptr, row_indices, values).run();
}
