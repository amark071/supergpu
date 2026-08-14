#include "io_func.hpp"

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef BINARY_CSC_SMOKE_PATH
#define BINARY_CSC_SMOKE_PATH "binary_csc_io_smoke.bin"
#endif

namespace {

template <class T>
void writeItems(std::ofstream& file, const T* values, std::size_t count)
{
    file.write(
        reinterpret_cast<const char*>(values),
        static_cast<std::streamsize>(count * sizeof(T)));
}

void writeBinaryCsc(
    const std::string& filename,
    bool floating_nnz,
    int index_base)
{
    const std::int8_t sparse = 1;
    const std::int32_t columns = 3;
    const std::int32_t rows = 3;
    const std::int64_t integer_nnz = 7;
    const double double_nnz = 7.0;
    const std::int64_t col_ptr[] = {
        index_base, 2 + index_base, 5 + index_base, 7 + index_base};
    const std::int32_t row_indices[] = {
        0 + index_base, 1 + index_base,
        0 + index_base, 1 + index_base, 2 + index_base,
        1 + index_base, 2 + index_base};
    const double values[] = {4.0, 0.75, 0.75, 3.0, 5.0, 5.0, 2.0};

    std::ofstream file(filename, std::ios::binary);
    writeItems(file, &sparse, 1);
    writeItems(file, &columns, 1);
    writeItems(file, &rows, 1);
    if (floating_nnz) {
        writeItems(file, &double_nnz, 1);
    } else {
        writeItems(file, &integer_nnz, 1);
    }
    writeItems(file, col_ptr, 4);
    writeItems(file, row_indices, 7);
    writeItems(file, values, 7);
    if (!file) {
        throw std::runtime_error("failed to create binary CSC smoke input");
    }
}

bool readAndCheck(const std::string& filename)
{
    int n = 0;
    int nnz = 0;
    std::vector<int> row_indices;
    std::vector<int> col_ptr;
    std::vector<float> values;
    readBinaryCSCMatrix(
        filename, n, nnz, row_indices, col_ptr, values);
    const std::vector<int> expected_ptr = {0, 2, 5, 7};
    const std::vector<int> expected_rows = {0, 1, 0, 1, 2, 1, 2};
    const std::vector<float> expected_values = {
        4.0f, 0.75f, 0.75f, 3.0f, 5.0f, 5.0f, 2.0f};
    return n == 3 && nnz == 7 && col_ptr == expected_ptr &&
        row_indices == expected_rows && values == expected_values;
}

} // namespace

int main()
{
    const std::string double_path = BINARY_CSC_SMOKE_PATH;
    const std::string integer_path = double_path + ".int64";
    try {
        writeBinaryCsc(double_path, true, 0);
        writeBinaryCsc(integer_path, false, 1);
        const bool valid = readAndCheck(double_path) && readAndCheck(integer_path);
        std::remove(double_path.c_str());
        std::remove(integer_path.c_str());
        if (!valid) {
            std::cerr << "binary CSC conversion mismatch\n";
            return 1;
        }
    } catch (...) {
        std::remove(double_path.c_str());
        std::remove(integer_path.c_str());
        throw;
    }
    std::cout << "binary CSC int64/double header smoke test passed\n";
    return 0;
}
