#include "cholmod_symbolic.hpp"
#include "gpu_supernodal_ldlt.hpp"
#include "io_func.hpp"
#include "ordering.hpp"

#include <iostream>
#include <string>
#include <vector>

#ifndef SUPERNODAL_VISUALIZATION_DIR
#define SUPERNODAL_VISUALIZATION_DIR "."
#endif

int main()
{
    // 读入后，矩阵在内存中统一使用0-based CSC格式。
    std::vector<int> A_rows;
    std::vector<int> A_cols;
    std::vector<float> A_val;
    int A_n, A_nnz;
    readCSCMatrix("data/A_1215.dat", A_n, A_nnz, A_rows, A_cols, A_val);

    // 分别测试METIS和AMD，选择符号填充元更少的排列。
    const OrderingResult ordering = computeBestOrdering(A_n, A_cols, A_rows);

    std::vector<int> ordered_rows;
    std::vector<int> ordered_cols;
    std::vector<float> ordered_values;
    applySymmetricPermutationCSC(A_n, A_cols, A_rows, A_val, ordering.iperm, ordered_cols, ordered_rows, ordered_values);

    writeCSCMatrix("data/A_1215_ordered.dat", A_n, ordered_rows, ordered_cols, ordered_values);
    std::cout << "Loaded " << A_val.size() << " nonzero entries" << '\n';
    std::cout << A_n << "x" << A_n << ", nonzero = " << A_nnz << '\n';
    std::cout << "METIS symbolic fill-in = " << ordering.metis_fill.fill_entries << ", ratio = " << ordering.metis_fill.fill_ratio << '\n';
    std::cout << "AMD symbolic fill-in = " << ordering.amd_fill.fill_entries << ", ratio = " << ordering.amd_fill.fill_ratio << '\n';
    std::cout << "Selected ordering method = " << orderingMethodName(ordering.method) << '\n';
    std::cout << "Selected symbolic factor L nonzeros = "<< ordering.selected_fill.factor_nonzeros << '\n';
    std::cout << "Reordered nonzeros = " << ordered_values.size() << '\n';

    // 符号分析
    const CholmodSymbolicResult symbolic = analyzeBasicSupernodesWithCholmod(A_n, ordered_cols, ordered_rows);
    writeSymbolicVisualizationData(SUPERNODAL_VISUALIZATION_DIR, A_n, symbolic);
    std::cout << "Elimination tree columns = " << symbolic.column_parent.size() << '\n';
    std::cout << "Basic supernodes = " << symbolic.supernode_parent.size() << '\n';
    std::cout << "Visualization data = " << SUPERNODAL_VISUALIZATION_DIR << '\n';

    // 数值分解：节点分类依据实际候选宽度（包含子节点上传的延迟列）。
    // 1 列节点与 2--64 列节点分别批处理；更大的节点使用 64 列 BK panel。
    GpuLdltOptions gpu_options;
    gpu_options.batched_width_limit = 64;
    gpu_options.panel_size = 64;
    gpu_options.max_batch_nodes = 256;

    GpuSupernodalLdltFactor gpu_factor(gpu_options);
    const GpuLdltStatistics gpu_statistics = gpu_factor.factorize(
        A_n, ordered_cols, ordered_rows, ordered_values, symbolic);

    std::cout << "GPU single-column nodes = "
              << gpu_statistics.single_column_nodes << '\n';
    std::cout << "GPU 2--64 column nodes = "
              << gpu_statistics.batched_small_medium_nodes << '\n';
    std::cout << "GPU >64 column nodes = "
              << gpu_statistics.large_panel_nodes << '\n';
    std::cout << "Accepted 1x1 pivots = "
              << gpu_statistics.accepted_one_by_one_pivots << '\n';
    std::cout << "Accepted 2x2 pivot blocks = "
              << gpu_statistics.accepted_two_by_two_pivots << '\n';
    std::cout << "Delayed pivot events = "
              << gpu_statistics.delayed_columns << '\n';
    std::cout << "Unresolved root columns = "
              << gpu_statistics.unresolved_root_columns << '\n';
    std::cout << "Supernode tree waves = "
              << gpu_statistics.tree_waves << '\n';
    std::cout << "Maximum input asymmetry = "
              << gpu_statistics.maximum_input_asymmetry << '\n';
    std::cout << "GPU factorization time (ms) = "
              << gpu_statistics.factorization_milliseconds << '\n';
    std::cout << "GPU LDLT status = " << gpu_factor.diagnostic() << '\n';
    
    return gpu_factor.complete() ? 0 : 2;
}
