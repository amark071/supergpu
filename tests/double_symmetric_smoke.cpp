#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef DOUBLE_SYMMETRIC_SMOKE_PATH
#define DOUBLE_SYMMETRIC_SMOKE_PATH "double_symmetric_smoke.bin"
#endif

int runDoubleSymmetricSolver(
    const std::string& filename,
    bool input_is_binary_csc);

namespace {

template <class T>
void writeItems(std::ofstream& file, const T* values, std::size_t count)
{
    file.write(
        reinterpret_cast<const char*>(values),
        static_cast<std::streamsize>(count * sizeof(T)));
}

void writePositiveDefiniteBinaryCsc(const std::string& filename)
{
    const std::int8_t sparse = 1;
    const std::int32_t columns = 4;
    const std::int32_t rows = 4;
    const double nnz = 10.0;
    const std::int64_t col_ptr[] = {0, 2, 5, 8, 10};
    const std::int32_t row_indices[] = {
        0, 1,
        0, 1, 2,
        1, 2, 3,
        2, 3};
    const double values[] = {
        4.0, 1.0,
        1.0, 4.0, 1.0,
        1.0, 4.0, 1.0,
        1.0, 3.0};

    std::ofstream file(filename.c_str(), std::ios::binary);
    writeItems(file, &sparse, 1);
    writeItems(file, &columns, 1);
    writeItems(file, &rows, 1);
    writeItems(file, &nnz, 1);
    writeItems(file, col_ptr, 5);
    writeItems(file, row_indices, 10);
    writeItems(file, values, 10);
    if (!file) {
        throw std::runtime_error("failed to create FP64 CSC smoke input");
    }
}

} // namespace

int main()
{
    const std::string path = DOUBLE_SYMMETRIC_SMOKE_PATH;
    try {
        writePositiveDefiniteBinaryCsc(path);
        const int status = runDoubleSymmetricSolver(path, true);
        std::remove(path.c_str());
        if (status != 0) {
            std::cerr << "FP64 symmetric solver returned " << status << '\n';
            return status;
        }
    } catch (...) {
        std::remove(path.c_str());
        throw;
    }
    return 0;
}
