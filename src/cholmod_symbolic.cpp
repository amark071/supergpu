#include "cholmod_symbolic.hpp"

#include <cholmod.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class CholmodContext {
public:
    CholmodContext()
    {
        if (!cholmod_start(&common_)) {
            throw std::runtime_error("failed to initialize CHOLMOD");
        }
    }

    ~CholmodContext()
    {
        cholmod_finish(&common_);
    }

    cholmod_common* get()
    {
        return &common_;
    }

private:
    cholmod_common common_;
};

struct SparseDeleter {
    cholmod_common* common;

    void operator()(cholmod_sparse* matrix) const
    {
        if (matrix != 0) {
            cholmod_free_sparse(&matrix, common);
        }
    }
};

struct FactorDeleter {
    cholmod_common* common;

    void operator()(cholmod_factor* factor) const
    {
        if (factor != 0) {
            cholmod_free_factor(&factor, common);
        }
    }
};

void validateCSC(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices)
{
    if (n < 0 || col_ptr.size() != static_cast<std::size_t>(n + 1)) {
        throw std::invalid_argument("invalid CSC dimensions for CHOLMOD analysis");
    }
    if (col_ptr.empty() || col_ptr.front() != 0 ||
        col_ptr.back() != static_cast<int>(row_indices.size())) {
        throw std::invalid_argument("CSC column pointers must be zero-based");
    }

    for (int col = 0; col < n; ++col) {
        const int begin = col_ptr[static_cast<std::size_t>(col)];
        const int end = col_ptr[static_cast<std::size_t>(col + 1)];
        if (begin < 0 || end < begin ||
            end > static_cast<int>(row_indices.size())) {
            throw std::invalid_argument("invalid CSC column pointer range");
        }
        for (int p = begin; p < end; ++p) {
            const int row = row_indices[static_cast<std::size_t>(p)];
            if (row < 0 || row >= n) {
                throw std::invalid_argument("CSC row index is out of range");
            }
        }
    }
}

struct SymmetricUpperPattern {
    std::vector<int> col_ptr;
    std::vector<int> rows;
    std::size_t diagonal_nonzeros;
};

SymmetricUpperPattern buildSymmetricUpperPattern(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices)
{
    SymmetricUpperPattern pattern;
    pattern.col_ptr.assign(static_cast<std::size_t>(n + 1), 0);
    pattern.diagonal_nonzeros = 0;

    // The project input is normally a sorted, unique, explicitly symmetric
    // CSC matrix. In that common case one stored triangle is already the exact
    // CHOLMOD pattern: select it in two linear passes and avoid sorting and
    // deduplicating millions of mirrored entries.
    bool sorted_unique = true;
    std::size_t strict_upper = 0;
    std::size_t strict_lower = 0;
    for (int col = 0; col < n; ++col) {
        int previous = -1;
        const int begin = col_ptr[static_cast<std::size_t>(col)];
        const int end = col_ptr[static_cast<std::size_t>(col + 1)];
        for (int p = begin; p < end; ++p) {
            const int row = row_indices[static_cast<std::size_t>(p)];
            sorted_unique = sorted_unique && row > previous;
            previous = row;
            if (row < col) {
                ++strict_upper;
            } else if (row > col) {
                ++strict_lower;
            }
        }
    }
    bool exact_symmetric_pattern = false;
    if (sorted_unique && strict_upper == strict_lower && strict_upper != 0) {
        std::vector<int> transpose_ptr(static_cast<std::size_t>(n + 1), 0);
        for (int col = 0; col < n; ++col) {
            const int begin = col_ptr[static_cast<std::size_t>(col)];
            const int end = col_ptr[static_cast<std::size_t>(col + 1)];
            for (int p = begin; p < end; ++p) {
                const int row = row_indices[static_cast<std::size_t>(p)];
                if (row > col) {
                    ++transpose_ptr[static_cast<std::size_t>(row + 1)];
                }
            }
        }
        for (int col = 0; col < n; ++col) {
            transpose_ptr[static_cast<std::size_t>(col + 1)] +=
                transpose_ptr[static_cast<std::size_t>(col)];
        }
        std::vector<int> transpose_rows(strict_lower);
        std::vector<int> transpose_next = transpose_ptr;
        for (int col = 0; col < n; ++col) {
            const int begin = col_ptr[static_cast<std::size_t>(col)];
            const int end = col_ptr[static_cast<std::size_t>(col + 1)];
            for (int p = begin; p < end; ++p) {
                const int row = row_indices[static_cast<std::size_t>(p)];
                if (row > col) {
                    transpose_rows[static_cast<std::size_t>(
                        transpose_next[static_cast<std::size_t>(row)]++)] = col;
                }
            }
        }
        exact_symmetric_pattern = true;
        for (int col = 0; col < n && exact_symmetric_pattern; ++col) {
            int transpose_position =
                transpose_ptr[static_cast<std::size_t>(col)];
            const int transpose_end =
                transpose_ptr[static_cast<std::size_t>(col + 1)];
            const int begin = col_ptr[static_cast<std::size_t>(col)];
            const int end = col_ptr[static_cast<std::size_t>(col + 1)];
            for (int p = begin; p < end; ++p) {
                const int row = row_indices[static_cast<std::size_t>(p)];
                if (row < col &&
                    (transpose_position >= transpose_end ||
                     transpose_rows[static_cast<std::size_t>(
                         transpose_position++)] != row)) {
                    exact_symmetric_pattern = false;
                    break;
                }
            }
            exact_symmetric_pattern = exact_symmetric_pattern &&
                transpose_position == transpose_end;
        }
    }
    const bool direct_upper = sorted_unique &&
        (strict_lower == 0 || exact_symmetric_pattern);
    if (direct_upper) {
        for (int col = 0; col < n; ++col) {
            const int begin = col_ptr[static_cast<std::size_t>(col)];
            const int end = col_ptr[static_cast<std::size_t>(col + 1)];
            for (int p = begin; p < end; ++p) {
                if (row_indices[static_cast<std::size_t>(p)] <= col) {
                    ++pattern.col_ptr[static_cast<std::size_t>(col + 1)];
                }
            }
        }
        for (int col = 0; col < n; ++col) {
            pattern.col_ptr[static_cast<std::size_t>(col + 1)] +=
                pattern.col_ptr[static_cast<std::size_t>(col)];
        }
        pattern.rows.resize(static_cast<std::size_t>(pattern.col_ptr.back()));
        std::size_t destination = 0;
        for (int col = 0; col < n; ++col) {
            const int begin = col_ptr[static_cast<std::size_t>(col)];
            const int end = col_ptr[static_cast<std::size_t>(col + 1)];
            for (int p = begin; p < end; ++p) {
                const int row = row_indices[static_cast<std::size_t>(p)];
                if (row <= col) {
                    pattern.rows[destination++] = row;
                    if (row == col) {
                        ++pattern.diagonal_nonzeros;
                    }
                }
            }
        }
        return pattern;
    }

    for (int col = 0; col < n; ++col) {
        const int begin = col_ptr[static_cast<std::size_t>(col)];
        const int end = col_ptr[static_cast<std::size_t>(col + 1)];
        for (int p = begin; p < end; ++p) {
            const int row = row_indices[static_cast<std::size_t>(p)];
            const int upper_col = std::max(row, col);
            ++pattern.col_ptr[static_cast<std::size_t>(upper_col + 1)];
        }
    }

    for (int col = 0; col < n; ++col) {
        pattern.col_ptr[static_cast<std::size_t>(col + 1)] +=
            pattern.col_ptr[static_cast<std::size_t>(col)];
    }
    pattern.rows.resize(static_cast<std::size_t>(pattern.col_ptr.back()));
    std::vector<int> next = pattern.col_ptr;
    for (int col = 0; col < n; ++col) {
        const int begin = col_ptr[static_cast<std::size_t>(col)];
        const int end = col_ptr[static_cast<std::size_t>(col + 1)];
        for (int p = begin; p < end; ++p) {
            const int row = row_indices[static_cast<std::size_t>(p)];
            const int upper_col = std::max(row, col);
            pattern.rows[static_cast<std::size_t>(
                next[static_cast<std::size_t>(upper_col)]++)] =
                std::min(row, col);
        }
    }

    // Sort and compact duplicate lower/upper copies in place. Every compacted
    // destination precedes its source, so a simple forward copy is safe.
    std::vector<int> compact_ptr(static_cast<std::size_t>(n + 1), 0);
    std::size_t destination = 0;
    for (int col = 0; col < n; ++col) {
        const int begin = pattern.col_ptr[static_cast<std::size_t>(col)];
        const int end = pattern.col_ptr[static_cast<std::size_t>(col + 1)];
        std::sort(pattern.rows.begin() + begin, pattern.rows.begin() + end);
        int previous = -1;
        for (int position = begin; position < end; ++position) {
            const int row = pattern.rows[static_cast<std::size_t>(position)];
            if (row != previous) {
                pattern.rows[destination++] = row;
                previous = row;
                if (row == col) {
                    ++pattern.diagonal_nonzeros;
                }
            }
        }
        compact_ptr[static_cast<std::size_t>(col + 1)] =
            static_cast<int>(destination);
    }
    pattern.rows.resize(destination);
    pattern.col_ptr.swap(compact_ptr);
    return pattern;
}

std::ofstream openOutputFile(
    const std::string& output_directory,
    const std::string& filename)
{
    const std::string path = output_directory + "/" + filename;
    std::ofstream file(path.c_str());
    if (!file) {
        throw std::runtime_error("could not create visualization data file: " + path);
    }
    return file;
}

} // namespace

CholmodSymbolicResult analyzeWithCholmodOrdering(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices,
    int ordering)
{
    validateCSC(n, col_ptr, row_indices);
    SymmetricUpperPattern upper_pattern;
    if (ordering == CHOLMOD_NATURAL) {
        // This path receives the already permuted CSC matrix. Keeping the
        // natural-order compatibility behavior fully canonicalizes it.
        upper_pattern = buildSymmetricUpperPattern(n, col_ptr, row_indices);
    } else {
        // Fast combined ordering path. The project input is sorted and stores
        // the full symmetric pattern, so its upper triangle is already the
        // exact CHOLMOD structure. Avoid the expensive general dedup fallback.
        upper_pattern.col_ptr.assign(static_cast<std::size_t>(n + 1), 0);
        upper_pattern.diagonal_nonzeros = 0;
        int previous_end = 0;
        for (int col = 0; col < n; ++col) {
            int previous = -1;
            const int begin = col_ptr[static_cast<std::size_t>(col)];
            const int end = col_ptr[static_cast<std::size_t>(col + 1)];
            if (begin != previous_end) {
                throw std::invalid_argument("CSC column pointers are not monotonic");
            }
            for (int p = begin; p < end; ++p) {
                const int row = row_indices[static_cast<std::size_t>(p)];
                if (row <= previous) {
                    throw std::invalid_argument(
                        "fast symbolic path requires sorted unique CSC rows");
                }
                previous = row;
                if (row <= col) {
                    ++upper_pattern.col_ptr[static_cast<std::size_t>(col + 1)];
                }
            }
            previous_end = end;
        }
        for (int col = 0; col < n; ++col) {
            upper_pattern.col_ptr[static_cast<std::size_t>(col + 1)] +=
                upper_pattern.col_ptr[static_cast<std::size_t>(col)];
        }
        upper_pattern.rows.resize(
            static_cast<std::size_t>(upper_pattern.col_ptr.back()));
        std::size_t destination = 0;
        for (int col = 0; col < n; ++col) {
            const int begin = col_ptr[static_cast<std::size_t>(col)];
            const int end = col_ptr[static_cast<std::size_t>(col + 1)];
            for (int p = begin; p < end; ++p) {
                const int row = row_indices[static_cast<std::size_t>(p)];
                if (row <= col) {
                    upper_pattern.rows[destination++] = row;
                    if (row == col) {
                        ++upper_pattern.diagonal_nonzeros;
                    }
                }
            }
        }
    }

    const std::size_t upper_nnz = upper_pattern.rows.size();

    CholmodContext context;
    cholmod_common* common = context.get();

    // 输入已经由项目中的 AMD/METIS 重排，CHOLMOD 不再改变列编号。
    common->nmethods = 1;
    common->method[0].ordering = ordering;
    common->postorder = 0;
    common->supernodal = CHOLMOD_SUPERNODAL;
    
    // 合并后列数不超过 4：直接允许；
    // 不超过 16：只要新增结构零比例低于 80%；
    // 不超过 48：要求新增结构零比例低于 10%；
    // 更大的超节点：要求新增结构零比例低于 5%。
    common->nrelax[0] = 4;
    common->nrelax[1] = 16;
    common->nrelax[2] = 48;

    common->zrelax[0] = 0.8;
    common->zrelax[1] = 0.1;
    common->zrelax[2] = 0.05;

    std::unique_ptr<cholmod_sparse, SparseDeleter> matrix(
        cholmod_allocate_sparse(
            static_cast<std::size_t>(n),
            static_cast<std::size_t>(n),
            upper_nnz,
            1,
            1,
            1,
            CHOLMOD_PATTERN,
            common),
        SparseDeleter{common});

    if (!matrix) {
        throw std::runtime_error("CHOLMOD could not allocate the symbolic matrix");
    }

    std::int32_t* cholmod_col_ptr =
        static_cast<std::int32_t*>(matrix->p);
    std::int32_t* cholmod_rows =
        static_cast<std::int32_t*>(matrix->i);

    for (int col = 0; col <= n; ++col) {
        cholmod_col_ptr[static_cast<std::size_t>(col)] =
            static_cast<std::int32_t>(
                upper_pattern.col_ptr[static_cast<std::size_t>(col)]);
    }
    for (std::size_t position = 0;
         position < upper_pattern.rows.size(); ++position) {
        cholmod_rows[position] =
            static_cast<std::int32_t>(upper_pattern.rows[position]);
    }

    std::unique_ptr<cholmod_factor, FactorDeleter> factor(
        cholmod_analyze(matrix.get(), common),
        FactorDeleter{common});
    if (!factor) {
        throw std::runtime_error("CHOLMOD symbolic analysis failed");
    }
    if (!factor->is_super) {
        throw std::runtime_error("CHOLMOD did not return a supernodal symbolic factor");
    }

    CholmodSymbolicResult result;
    result.input_off_diagonal_nonzeros =
        upper_nnz - upper_pattern.diagonal_nonzeros;
    result.perm.resize(static_cast<std::size_t>(n));
    result.iperm.resize(static_cast<std::size_t>(n));
    const bool long_indices = factor->itype == CHOLMOD_LONG;
    const std::int32_t* factor_perm_int =
        long_indices ? 0 : static_cast<const std::int32_t*>(factor->Perm);
    const std::int64_t* factor_perm_long =
        long_indices ? static_cast<const std::int64_t*>(factor->Perm) : 0;
    for (int new_index = 0; new_index < n; ++new_index) {
        const int old_index = ordering == CHOLMOD_NATURAL
            ? new_index
            : (long_indices
                ? static_cast<int>(factor_perm_long[new_index])
                : static_cast<int>(factor_perm_int[new_index]));
        result.perm[static_cast<std::size_t>(new_index)] = old_index;
        result.iperm[static_cast<std::size_t>(old_index)] = new_index;
    }

    result.column_count.resize(static_cast<std::size_t>(n));
    if (long_indices) {
        const std::int64_t* column_count =
            static_cast<const std::int64_t*>(factor->ColCount);
        for (int col = 0; col < n; ++col) {
            result.column_count[static_cast<std::size_t>(col)] =
                static_cast<int>(column_count[col]);
        }
    } else {
        const std::int32_t* column_count =
            static_cast<const std::int32_t*>(factor->ColCount);
        result.column_count.assign(column_count, column_count + n);
    }

    const std::size_t supernode_count = factor->nsuper;
    result.supernode_ptr.resize(supernode_count + 1);
    result.row_ptr.resize(supernode_count + 1);
    if (long_indices) {
        const std::int64_t* super =
            static_cast<const std::int64_t*>(factor->super);
        const std::int64_t* pi =
            static_cast<const std::int64_t*>(factor->pi);
        for (std::size_t entry = 0; entry <= supernode_count; ++entry) {
            result.supernode_ptr[entry] = static_cast<int>(super[entry]);
            result.row_ptr[entry] = static_cast<int>(pi[entry]);
        }
        const std::int64_t* rows =
            static_cast<const std::int64_t*>(factor->s);
        result.supernode_rows.resize(
            static_cast<std::size_t>(result.row_ptr.back()));
        for (std::size_t entry = 0;
             entry < result.supernode_rows.size(); ++entry) {
            result.supernode_rows[entry] = static_cast<int>(rows[entry]);
        }
    } else {
        const std::int32_t* super =
            static_cast<const std::int32_t*>(factor->super);
        const std::int32_t* pi =
            static_cast<const std::int32_t*>(factor->pi);
        const std::int32_t* rows =
            static_cast<const std::int32_t*>(factor->s);
        result.supernode_ptr.assign(super, super + supernode_count + 1);
        result.row_ptr.assign(pi, pi + supernode_count + 1);
        result.supernode_rows.assign(rows, rows + result.row_ptr.back());
    }

    // For a supernode, consecutive columns form a chain. The parent of its
    // last column is the first row below the dense diagonal block. This uses
    // the structure already produced by cholmod_analyze and avoids another
    // complete elimination-tree pass.
    result.column_parent.assign(static_cast<std::size_t>(n), -1);
    for (std::size_t s = 0; s < supernode_count; ++s) {
        const int first_col = result.supernode_ptr[s];
        const int end_col = result.supernode_ptr[s + 1];
        for (int col = first_col; col + 1 < end_col; ++col) {
            result.column_parent[static_cast<std::size_t>(col)] = col + 1;
        }
        const int first_update =
            result.row_ptr[s] + (end_col - first_col);
        if (end_col > first_col && first_update < result.row_ptr[s + 1]) {
            result.column_parent[static_cast<std::size_t>(end_col - 1)] =
                result.supernode_rows[static_cast<std::size_t>(first_update)];
        }
    }

    std::vector<int> column_to_supernode(static_cast<std::size_t>(n), -1);
    for (std::size_t s = 0; s < supernode_count; ++s) {
        for (int col = result.supernode_ptr[s];
             col < result.supernode_ptr[s + 1]; ++col) {
            column_to_supernode[static_cast<std::size_t>(col)] =
                static_cast<int>(s);
        }
    }

    result.supernode_parent.assign(supernode_count, -1);
    for (std::size_t s = 0; s < supernode_count; ++s) {
        const int last_col = result.supernode_ptr[s + 1] - 1;
        const int parent_col = result.column_parent[static_cast<std::size_t>(last_col)];
        if (parent_col >= 0) {
            result.supernode_parent[s] =
                column_to_supernode[static_cast<std::size_t>(parent_col)];
        }
    }

    return result;
}

CholmodSymbolicResult analyzeBasicSupernodesWithCholmod(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices)
{
    return analyzeWithCholmodOrdering(
        n, col_ptr, row_indices, CHOLMOD_NATURAL);
}

CholmodSymbolicResult analyzeAndOrderBasicSupernodesWithCholmod(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices)
{
    return analyzeWithCholmodOrdering(
        n, col_ptr, row_indices, CHOLMOD_METIS);
}

void writeSymbolicVisualizationData(
    const std::string& output_directory,
    int n,
    const CholmodSymbolicResult& symbolic)
{
    if (n < 0 ||
        symbolic.column_parent.size() != static_cast<std::size_t>(n) ||
        symbolic.column_count.size() != static_cast<std::size_t>(n) ||
        symbolic.supernode_ptr.empty() ||
        symbolic.supernode_ptr.front() != 0 ||
        symbolic.supernode_ptr.back() != n ||
        symbolic.row_ptr.size() != symbolic.supernode_ptr.size() ||
        symbolic.supernode_parent.size() + 1 != symbolic.supernode_ptr.size()) {
        throw std::invalid_argument("invalid symbolic result for visualization");
    }

    const std::size_t supernode_count = symbolic.supernode_parent.size();
    std::vector<int> column_to_supernode(static_cast<std::size_t>(n), -1);
    for (std::size_t s = 0; s < supernode_count; ++s) {
        for (int col = symbolic.supernode_ptr[s];
             col < symbolic.supernode_ptr[s + 1]; ++col) {
            column_to_supernode[static_cast<std::size_t>(col)] =
                static_cast<int>(s);
        }
    }

    std::ofstream columns = openOutputFile(output_directory, "column_tree.csv");
    columns << "column,parent,supernode,column_count\n";
    for (int col = 0; col < n; ++col) {
        columns << col << ','
                << symbolic.column_parent[static_cast<std::size_t>(col)] << ','
                << column_to_supernode[static_cast<std::size_t>(col)] << ','
                << symbolic.column_count[static_cast<std::size_t>(col)] << '\n';
    }

    std::ofstream supernodes = openOutputFile(output_directory, "supernodes.csv");
    supernodes << "supernode,first_col,end_col,parent,row_begin,row_end,row_count,width\n";
    for (std::size_t s = 0; s < supernode_count; ++s) {
        const int first_col = symbolic.supernode_ptr[s];
        const int end_col = symbolic.supernode_ptr[s + 1];
        const int row_begin = symbolic.row_ptr[s];
        const int row_end = symbolic.row_ptr[s + 1];
        supernodes << s << ','
                   << first_col << ','
                   << end_col << ','
                   << symbolic.supernode_parent[s] << ','
                   << row_begin << ','
                   << row_end << ','
                   << (row_end - row_begin) << ','
                   << (end_col - first_col) << '\n';
    }

    std::ofstream rows = openOutputFile(output_directory, "supernode_rows.csv");
    rows << "supernode,row\n";
    for (std::size_t s = 0; s < supernode_count; ++s) {
        for (int p = symbolic.row_ptr[s]; p < symbolic.row_ptr[s + 1]; ++p) {
            rows << s << ','
                 << symbolic.supernode_rows[static_cast<std::size_t>(p)] << '\n';
        }
    }
}
