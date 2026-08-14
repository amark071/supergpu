#include "io_func.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const std::uint64_t kBinaryHeaderBytes = 1U + 4U + 4U + 8U;
const std::size_t kReadChunkEntries = 1U << 20;

template <class T>
void readExact(
    std::ifstream& file,
    T* destination,
    std::size_t count,
    const char* field)
{
    if (count == 0) {
        return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::runtime_error(std::string(field) + " is too large to read");
    }
    const std::size_t bytes = count * sizeof(T);
    if (bytes > static_cast<std::size_t>(
                    std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error(std::string(field) + " is too large to read");
    }
    file.read(
        reinterpret_cast<char*>(destination),
        static_cast<std::streamsize>(bytes));
    if (!file) {
        throw std::runtime_error(
            std::string("truncated binary CSC while reading ") + field);
    }
}

std::uint64_t binaryFileSize(std::ifstream& file)
{
    file.seekg(0, std::ios::end);
    const std::streampos end = file.tellg();
    if (end < 0) {
        throw std::runtime_error("cannot determine binary CSC file size");
    }
    file.seekg(0, std::ios::beg);
    return static_cast<std::uint64_t>(end);
}

bool sizeMatches(
    std::int32_t columns,
    std::int64_t nonzeros,
    std::uint64_t file_size)
{
    if (columns < 0 || nonzeros < 0 ||
        nonzeros > std::numeric_limits<int>::max()) {
        return false;
    }
    const std::uint64_t column_count =
        static_cast<std::uint64_t>(columns) + 1U;
    const std::uint64_t column_bytes = column_count * sizeof(std::int64_t);
    const std::uint64_t fixed_bytes = kBinaryHeaderBytes + column_bytes;
    const std::uint64_t entry_bytes =
        sizeof(std::int32_t) + sizeof(double);
    const std::uint64_t count = static_cast<std::uint64_t>(nonzeros);
    if (count > (std::numeric_limits<std::uint64_t>::max() - fixed_bytes) /
                    entry_bytes) {
        return false;
    }
    return fixed_bytes + count * entry_bytes == file_size;
}

std::int64_t decodeNonzeroCount(
    std::uint64_t raw,
    std::int32_t columns,
    std::uint64_t file_size)
{
    std::int64_t integer_count = 0;
    double floating_count = 0.0;
    std::memcpy(&integer_count, &raw, sizeof(integer_count));
    std::memcpy(&floating_count, &raw, sizeof(floating_count));
    if (sizeMatches(columns, integer_count, file_size)) {
        return integer_count;
    }
    if (std::isfinite(floating_count) && floating_count >= 0.0 &&
        floating_count <= static_cast<double>(
            std::numeric_limits<int>::max()) &&
        std::floor(floating_count) == floating_count) {
        const std::int64_t converted =
            static_cast<std::int64_t>(floating_count);
        if (sizeMatches(columns, converted, file_size)) {
            return converted;
        }
    }
    throw std::runtime_error(
        "binary CSC nnz/header does not match the file length");
}

int readColumnPointers(
    std::ifstream& file,
    int n,
    int nnz,
    std::vector<int>& col_ptr)
{
    col_ptr.resize(static_cast<std::size_t>(n + 1));
    std::vector<std::int64_t> buffer(
        std::min<std::size_t>(kReadChunkEntries, col_ptr.size()));
    std::int64_t previous = -1;
    int index_base = -1;
    for (std::size_t offset = 0; offset < col_ptr.size();) {
        const std::size_t count = std::min(buffer.size(), col_ptr.size() - offset);
        readExact(file, buffer.data(), count, "column pointers");
        for (std::size_t i = 0; i < count; ++i) {
            const std::int64_t raw = buffer[i];
            if (offset == 0 && i == 0) {
                if (raw != 0 && raw != 1) {
                    throw std::runtime_error(
                        "binary CSC column pointers must start at 0 or 1");
                }
                index_base = static_cast<int>(raw);
            }
            const std::int64_t normalized = raw - index_base;
            if (normalized < previous || normalized < 0 || normalized > nnz) {
                throw std::runtime_error(
                    "binary CSC column pointers are invalid or non-monotonic");
            }
            col_ptr[offset + i] = static_cast<int>(normalized);
            previous = normalized;
        }
        offset += count;
    }
    if (col_ptr.back() != nnz) {
        throw std::runtime_error("binary CSC final column pointer differs from nnz");
    }
    return index_base;
}

void readRowIndices(
    std::ifstream& file,
    int n,
    int nnz,
    int index_base,
    std::vector<int>& row_indices)
{
    row_indices.resize(static_cast<std::size_t>(nnz));
    std::vector<std::int32_t> buffer(
        std::min<std::size_t>(kReadChunkEntries, row_indices.size()));
    for (std::size_t offset = 0; offset < row_indices.size();) {
        const std::size_t count =
            std::min(buffer.size(), row_indices.size() - offset);
        readExact(file, buffer.data(), count, "row indices");
        for (std::size_t i = 0; i < count; ++i) {
            const std::int64_t normalized =
                static_cast<std::int64_t>(buffer[i]) - index_base;
            if (normalized < 0 || normalized >= n) {
                throw std::runtime_error("binary CSC row index is out of range");
            }
            row_indices[offset + i] = static_cast<int>(normalized);
        }
        offset += count;
    }
}

void readValues(
    std::ifstream& file,
    int nnz,
    std::vector<float>& values)
{
    values.resize(static_cast<std::size_t>(nnz));
    std::vector<double> buffer(
        std::min<std::size_t>(kReadChunkEntries, values.size()));
    const double float_max = std::numeric_limits<float>::max();
    for (std::size_t offset = 0; offset < values.size();) {
        const std::size_t count = std::min(buffer.size(), values.size() - offset);
        readExact(file, buffer.data(), count, "matrix values");
        for (std::size_t i = 0; i < count; ++i) {
            if (!std::isfinite(buffer[i]) || std::fabs(buffer[i]) > float_max) {
                throw std::runtime_error(
                    "binary CSC value cannot be represented as finite FP32");
            }
            values[offset + i] = static_cast<float>(buffer[i]);
        }
        offset += count;
    }
}

} // namespace

void readBinaryCSCMatrix(
    const std::string& filename,
    int& n,
    int& nnz,
    std::vector<int>& row_indices,
    std::vector<int>& col_ptr,
    std::vector<float>& values)
{
    static_assert(sizeof(std::int8_t) == 1, "int8_t must be one byte");
    static_assert(sizeof(std::int32_t) == 4, "int32_t must be four bytes");
    static_assert(sizeof(std::int64_t) == 8, "int64_t must be eight bytes");
    static_assert(sizeof(double) == 8, "double must be eight bytes");

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open input matrix file: " + filename);
    }
    const std::uint64_t file_size = binaryFileSize(file);
    std::int8_t sparse = 0;
    std::int32_t columns = 0;
    std::int32_t rows = 0;
    std::uint64_t raw_nnz = 0;
    readExact(file, &sparse, 1, "sparse flag");
    readExact(file, &columns, 1, "column count");
    readExact(file, &rows, 1, "row count");
    readExact(file, &raw_nnz, 1, "nonzero count");
    if (sparse != 1) {
        throw std::runtime_error("binary matrix is not marked as sparse CSC");
    }
    if (columns <= 0 || rows != columns) {
        throw std::runtime_error("binary CSC matrix must be nonempty and square");
    }
    const std::int64_t decoded_nnz =
        decodeNonzeroCount(raw_nnz, columns, file_size);
    const int parsed_n = static_cast<int>(columns);
    const int parsed_nnz = static_cast<int>(decoded_nnz);
    std::vector<int> parsed_col_ptr;
    std::vector<int> parsed_row_indices;
    std::vector<float> parsed_values;
    const int index_base = readColumnPointers(
        file, parsed_n, parsed_nnz, parsed_col_ptr);
    readRowIndices(
        file, parsed_n, parsed_nnz, index_base, parsed_row_indices);
    readValues(file, parsed_nnz, parsed_values);

    n = parsed_n;
    nnz = parsed_nnz;
    col_ptr.swap(parsed_col_ptr);
    row_indices.swap(parsed_row_indices);
    values.swap(parsed_values);
}

void readCSCMatrix(
    const std::string& filename,
    int& n,
    int& nnz,
    std::vector<int>& row_indices,
    std::vector<int>& col_ptr,
    std::vector<float>& values)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open input matrix file: " + filename);
    }
    int row_count = 0;
    int col_count = 0;
    if (!(file >> row_count >> col_count) || row_count < 0 ||
        row_count != col_count) {
        throw std::runtime_error("input matrix must be a valid square matrix");
    }
    n = col_count;
    col_ptr.resize(static_cast<std::size_t>(n + 1));
    for (int i = 0; i <= n; ++i) {
        int one_based_pointer = 0;
        if (!(file >> one_based_pointer)) {
            throw std::runtime_error("failed to read CSC column pointers");
        }
        col_ptr[static_cast<std::size_t>(i)] = one_based_pointer - 1;
    }
    if (col_ptr.front() != 0 || col_ptr.back() < 0) {
        throw std::runtime_error("input CSC column pointers must be one-based");
    }
    nnz = col_ptr.back();
    row_indices.resize(static_cast<std::size_t>(nnz));
    for (int i = 0; i < nnz; ++i) {
        int one_based_row = 0;
        if (!(file >> one_based_row)) {
            throw std::runtime_error("failed to read CSC row indices");
        }
        row_indices[static_cast<std::size_t>(i)] = one_based_row - 1;
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
    int n,
    const std::vector<int>& row_indices,
    const std::vector<int>& col_ptr,
    const std::vector<float>& values)
{
    if (n < 0 || col_ptr.size() != static_cast<std::size_t>(n + 1)) {
        throw std::invalid_argument("invalid CSC dimensions");
    }
    if (row_indices.size() != values.size()) {
        throw std::invalid_argument("CSC row and value counts do not match");
    }
    if (col_ptr.empty() || col_ptr.front() != 0 ||
        col_ptr.back() != static_cast<int>(row_indices.size())) {
        throw std::invalid_argument("CSC column pointers must be zero-based");
    }
    for (int col = 0; col < n; ++col) {
        if (col_ptr[static_cast<std::size_t>(col)] >
            col_ptr[static_cast<std::size_t>(col + 1)]) {
            throw std::invalid_argument("CSC column pointers are not monotonic");
        }
    }
    for (std::size_t i = 0; i < row_indices.size(); ++i) {
        if (row_indices[i] < 0 || row_indices[i] >= n) {
            throw std::invalid_argument("CSC row index is out of range");
        }
    }
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open output matrix file: " + filename);
    }
    file << n << '\n' << n << '\n';
    for (std::size_t i = 0; i < col_ptr.size(); ++i) {
        file << col_ptr[i] + 1 << '\n';
    }
    for (std::size_t i = 0; i < row_indices.size(); ++i) {
        file << row_indices[i] + 1 << '\n';
    }
    file << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::size_t i = 0; i < values.size(); ++i) {
        file << values[i] << '\n';
    }
    if (!file) {
        throw std::runtime_error("failed while writing matrix file: " + filename);
    }
}
