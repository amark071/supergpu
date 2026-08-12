#include "io_func.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

void readCooMatrix(const std::string& filename,
                   std::vector<int>& rows,
                   std::vector<int>& cols,
                   std::vector<float>& vals
){
    std::ifstream file(filename);
    if (!file.is_open()) {
        return;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int r, c;
        float v;
        if (iss >> r >> c >> v) {
            rows.push_back(r - 1);
            cols.push_back(c - 1);
            vals.push_back(v);
        }
    }
    file.close();
}


void readCSCMatrix(
    const std::string& filename,
    int& N, int& nnz,
    std::vector<int>& row_indice,
    std::vector<int>& col_indice,
    std::vector<float>& values
){
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open input matrix file: " + filename);
    }

    int row_count = 0;
    int col_count = 0;
    if (!(file >> row_count >> col_count) || row_count < 0 || row_count != col_count) {
        throw std::runtime_error("input matrix must be a valid square matrix");
    }
    N = col_count;

    col_indice.resize(static_cast<std::size_t>(N + 1));
    for (int i = 0; i <= N; ++i) {
        int one_based_pointer = 0;
        if (!(file >> one_based_pointer)) {
            throw std::runtime_error("failed to read CSC column pointers");
        }
        col_indice[static_cast<std::size_t>(i)] = one_based_pointer - 1;
    }

    if (col_indice.front() != 0 || col_indice.back() < 0) {
        throw std::runtime_error("input CSC column pointers must be one-based");
    }
    nnz = col_indice.back();

    row_indice.resize(static_cast<std::size_t>(nnz));
    for (int i = 0; i < nnz; ++i) {
        int one_based_row = 0;
        if (!(file >> one_based_row)) {
            throw std::runtime_error("failed to read CSC row indices");
        }
        row_indice[static_cast<std::size_t>(i)] = one_based_row - 1;
    }

    values.resize(static_cast<std::size_t>(nnz));
    for (int i = 0; i < nnz; ++i) {
        if (!(file >> values[static_cast<std::size_t>(i)])) {
            throw std::runtime_error("failed to read CSC values");
        }
    }
}

void writeCSCMatrix(
    const std::string& filename,
    int N,
    const std::vector<int>& row_indice,
    const std::vector<int>& col_indice,
    const std::vector<float>& values
){
    if (N < 0 || col_indice.size() != static_cast<std::size_t>(N + 1)) {
        throw std::invalid_argument("invalid CSC dimensions");
    }
    if (row_indice.size() != values.size()) {
        throw std::invalid_argument("CSC row and value counts do not match");
    }
    if (col_indice.empty() || col_indice.front() != 0 ||
        col_indice.back() != static_cast<int>(row_indice.size())) {
        throw std::invalid_argument("CSC column pointers must be zero-based");
    }
    for (int col = 0; col < N; ++col) {
        if (col_indice[static_cast<std::size_t>(col)] >
            col_indice[static_cast<std::size_t>(col + 1)]) {
            throw std::invalid_argument("CSC column pointers are not monotonic");
        }
    }
    for (std::size_t i = 0; i < row_indice.size(); ++i) {
        if (row_indice[i] < 0 || row_indice[i] >= N) {
            throw std::invalid_argument("CSC row index is out of range");
        }
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open output matrix file: " + filename);
    }

    file << N << '\n' << N << '\n';
    for (std::size_t i = 0; i < col_indice.size(); ++i) {
        file << col_indice[i] + 1 << '\n';
    }
    for (std::size_t i = 0; i < row_indice.size(); ++i) {
        file << row_indice[i] + 1 << '\n';
    }

    file << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::size_t i = 0; i < values.size(); ++i) {
        file << values[i] << '\n';
    }

    if (!file) {
        throw std::runtime_error("failed while writing matrix file: " + filename);
    }
}

void csc_to_coo(
    int N, int nnz,
    const std::vector<int>& col_indice,     
    const std::vector<int>& row_indice,     
    const std::vector<float>& values_csc,  
    std::vector<int>& row_coo,       
    std::vector<int>& col_coo,        
    std::vector<float>& values_coo          
){
    row_coo.resize(nnz);
    col_coo.resize(nnz);
    values_coo.resize(nnz);

    int coo_idx = 0;
    
    // 遍历每一列
    for (int col = 0; col < N; ++col) {
        int start_idx = col_indice[col];
        int end_idx = col_indice[col + 1];
        
        // 遍历该列的所有非零元素
        for (int i = start_idx; i < end_idx; ++i) {
            row_coo[coo_idx] = row_indice[i];    
            col_coo[coo_idx] = col;
            values_coo[coo_idx] = values_csc[i];    // 值不变
            ++coo_idx;
        }
    }
}
