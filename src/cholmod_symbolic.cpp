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

std::vector<std::vector<int> > buildSymmetricUpperPattern(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices)
{
    std::vector<std::vector<int> > upper_columns(static_cast<std::size_t>(n));

    for (int col = 0; col < n; ++col) {
        const int begin = col_ptr[static_cast<std::size_t>(col)];
        const int end = col_ptr[static_cast<std::size_t>(col + 1)];
        for (int p = begin; p < end; ++p) {
            const int row = row_indices[static_cast<std::size_t>(p)];
            const int upper_col = std::max(row, col);
            const int upper_row = std::min(row, col);
            upper_columns[static_cast<std::size_t>(upper_col)].push_back(upper_row);
        }
    }

    for (int col = 0; col < n; ++col) {
        std::vector<int>& rows = upper_columns[static_cast<std::size_t>(col)];
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }

    return upper_columns;
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

CholmodSymbolicResult analyzeBasicSupernodesWithCholmod(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices)
{
    validateCSC(n, col_ptr, row_indices);
    const std::vector<std::vector<int> > upper_columns =
        buildSymmetricUpperPattern(n, col_ptr, row_indices);

    std::size_t upper_nnz = 0;
    for (int col = 0; col < n; ++col) {
        upper_nnz += upper_columns[static_cast<std::size_t>(col)].size();
    }

    CholmodContext context;
    cholmod_common* common = context.get();

    // 输入已经由项目中的 AMD/METIS 重排，CHOLMOD 不再改变列编号。
    common->nmethods = 1;
    common->method[0].ordering = CHOLMOD_NATURAL;
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

    std::size_t offset = 0;
    cholmod_col_ptr[0] = 0;
    for (int col = 0; col < n; ++col) {
        const std::vector<int>& rows = upper_columns[static_cast<std::size_t>(col)];
        for (std::size_t k = 0; k < rows.size(); ++k) {
            cholmod_rows[offset++] = static_cast<std::int32_t>(rows[k]);
        }
        cholmod_col_ptr[static_cast<std::size_t>(col + 1)] =
            static_cast<std::int32_t>(offset);
    }

    std::vector<std::int32_t> parent(static_cast<std::size_t>(n), -1);
    if (!cholmod_etree(matrix.get(), parent.data(), common)) {
        throw std::runtime_error("CHOLMOD failed to construct the elimination tree");
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
    result.column_parent.assign(parent.begin(), parent.end());

    const std::int32_t* column_count =
        static_cast<const std::int32_t*>(factor->ColCount);
    result.column_count.assign(column_count, column_count + n);

    const std::size_t supernode_count = factor->nsuper;
    const std::int32_t* super =
        static_cast<const std::int32_t*>(factor->super);
    const std::int32_t* pi =
        static_cast<const std::int32_t*>(factor->pi);
    const std::int32_t* rows =
        static_cast<const std::int32_t*>(factor->s);

    result.supernode_ptr.assign(super, super + supernode_count + 1);
    result.row_ptr.assign(pi, pi + supernode_count + 1);
    result.supernode_rows.assign(rows, rows + result.row_ptr.back());

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
