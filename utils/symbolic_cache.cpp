#include "symbolic_cache.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>

namespace {

const std::uint32_t kSymbolicCacheVersion = 1;

template <typename T>
bool readBinary(std::ifstream& input, T& value)
{
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(input);
}

template <typename T>
void writeBinary(std::ofstream& output, const T& value)
{
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

bool readIntVector(
    std::ifstream& input,
    std::vector<int>& values,
    std::uint64_t maximum_size)
{
    std::uint64_t size = 0;
    if (!readBinary(input, size) || size > maximum_size) {
        return false;
    }
    values.resize(static_cast<std::size_t>(size));
    if (size != 0) {
        input.read(
            reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(size * sizeof(int)));
    }
    return static_cast<bool>(input);
}

void writeIntVector(std::ofstream& output, const std::vector<int>& values)
{
    const std::uint64_t size = static_cast<std::uint64_t>(values.size());
    writeBinary(output, size);
    if (!values.empty()) {
        output.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(int)));
    }
}

bool hasValidSymbolicShape(int n, const CholmodSymbolicResult& symbolic)
{
    const std::size_t n_size = static_cast<std::size_t>(n);
    return symbolic.perm.size() == n_size &&
        symbolic.iperm.size() == n_size &&
        symbolic.column_parent.size() == n_size &&
        symbolic.column_count.size() == n_size &&
        !symbolic.supernode_ptr.empty() &&
        symbolic.supernode_ptr.front() == 0 &&
        symbolic.supernode_ptr.back() == n &&
        symbolic.supernode_parent.size() + 1 ==
            symbolic.supernode_ptr.size() &&
        symbolic.row_ptr.size() == symbolic.supernode_ptr.size() &&
        !symbolic.row_ptr.empty() && symbolic.row_ptr.front() == 0 &&
        symbolic.row_ptr.back() ==
            static_cast<int>(symbolic.supernode_rows.size());
}

} // namespace

std::uint64_t symbolicStructureHash(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices)
{
    std::uint64_t hash = 1469598103934665603ull;
    const std::uint64_t prime = 1099511628211ull;
    hash = (hash ^ static_cast<std::uint32_t>(n)) * prime;
    for (std::size_t i = 0; i < col_ptr.size(); ++i) {
        hash = (hash ^ static_cast<std::uint32_t>(col_ptr[i])) * prime;
    }
    for (std::size_t i = 0; i < row_indices.size(); ++i) {
        hash = (hash ^ static_cast<std::uint32_t>(row_indices[i])) * prime;
    }
    return hash;
}

bool loadSymbolicCache(
    const std::string& path,
    int n,
    std::size_t input_nonzeros,
    std::uint64_t structure_hash,
    CholmodSymbolicResult& symbolic)
{
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input) {
        return false;
    }

    char magic[8] = {};
    std::uint32_t version = 0;
    std::int32_t cached_n = 0;
    std::uint64_t cached_nonzeros = 0;
    std::uint64_t cached_hash = 0;
    std::uint64_t input_off_diagonal_nonzeros = 0;
    input.read(magic, sizeof(magic));
    if (!input || std::memcmp(magic, "SGLDLTSY", 8) != 0 ||
        !readBinary(input, version) ||
        !readBinary(input, cached_n) ||
        !readBinary(input, cached_nonzeros) ||
        !readBinary(input, cached_hash) ||
        !readBinary(input, input_off_diagonal_nonzeros) ||
        version != kSymbolicCacheVersion || cached_n != n ||
        cached_nonzeros != input_nonzeros || cached_hash != structure_hash) {
        return false;
    }

    const std::uint64_t n_size = static_cast<std::uint64_t>(n);
    const std::uint64_t maximum_factor_entries = n_size * (n_size + 1) / 2;
    if (!readIntVector(input, symbolic.perm, n_size) ||
        !readIntVector(input, symbolic.iperm, n_size) ||
        !readIntVector(input, symbolic.column_parent, n_size) ||
        !readIntVector(input, symbolic.column_count, n_size) ||
        !readIntVector(input, symbolic.supernode_ptr, n_size + 1) ||
        !readIntVector(input, symbolic.supernode_parent, n_size) ||
        !readIntVector(input, symbolic.row_ptr, n_size + 1) ||
        !readIntVector(
            input, symbolic.supernode_rows, maximum_factor_entries)) {
        return false;
    }
    symbolic.input_off_diagonal_nonzeros =
        static_cast<std::size_t>(input_off_diagonal_nonzeros);
    return hasValidSymbolicShape(n, symbolic);
}

void saveSymbolicCache(
    const std::string& path,
    int n,
    std::size_t input_nonzeros,
    std::uint64_t structure_hash,
    const CholmodSymbolicResult& symbolic)
{
    std::ofstream output(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!output) {
        std::cerr << "Warning: could not create symbolic cache: "
                  << path << '\n';
        return;
    }
    output.write("SGLDLTSY", 8);
    writeBinary(output, kSymbolicCacheVersion);
    writeBinary(output, static_cast<std::int32_t>(n));
    writeBinary(output, static_cast<std::uint64_t>(input_nonzeros));
    writeBinary(output, structure_hash);
    writeBinary(
        output,
        static_cast<std::uint64_t>(symbolic.input_off_diagonal_nonzeros));
    writeIntVector(output, symbolic.perm);
    writeIntVector(output, symbolic.iperm);
    writeIntVector(output, symbolic.column_parent);
    writeIntVector(output, symbolic.column_count);
    writeIntVector(output, symbolic.supernode_ptr);
    writeIntVector(output, symbolic.supernode_parent);
    writeIntVector(output, symbolic.row_ptr);
    writeIntVector(output, symbolic.supernode_rows);
}
