#ifndef SUPERNODAL_GPU_INCLUDE_CHOLMOD_SYMBOLIC_HPP
#define SUPERNODAL_GPU_INCLUDE_CHOLMOD_SYMBOLIC_HPP

#include <string>
#include <vector>

struct CholmodSymbolicResult {
    // column_parent[j] is the parent column of j; -1 denotes a root.
    std::vector<int> column_parent;

    // Symbolic nonzeros in each column of L, including the diagonal.
    std::vector<int> column_count;

    // Supernode s owns columns [supernode_ptr[s], supernode_ptr[s + 1]).
    std::vector<int> supernode_ptr;

    // supernode_parent[s] is the parent of s; -1 denotes a root.
    std::vector<int> supernode_parent;

    // Rows of s occupy supernode_rows[row_ptr[s] ... row_ptr[s + 1]).
    std::vector<int> row_ptr;
    std::vector<int> supernode_rows;
};

// Analyze a reordered, zero-based symmetric CSC structure with CHOLMOD.
CholmodSymbolicResult analyzeBasicSupernodesWithCholmod(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices);

void writeSymbolicVisualizationData(
    const std::string& output_directory,
    int n,
    const CholmodSymbolicResult& symbolic);

#endif
