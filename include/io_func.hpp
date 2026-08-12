#ifndef SUPERNODAL_GPU_INCLUDE_IO_FUNC_HPP
#define SUPERNODAL_GPU_INCLUDE_IO_FUNC_HPP

#include <string>
#include <vector>

/** @brief 读取1-based COO文件，并转换为0-based内存数据 */
void readCooMatrix(const std::string& filename,
                   std::vector<int>& rows,
                   std::vector<int>& cols,
                   std::vector<float>& vals);

/** @brief 读取1-based CSC文件，并转换为0-based内存数据 */
void readCSCMatrix(const std::string& filename,
                   int& n,
                   int& nnz,
                   std::vector<int>& row_indices,
                   std::vector<int>& col_ptr,
                   std::vector<float>& values);

/** @brief 将0-based CSC矩阵写成与A_1215.dat兼容的1-based文件 */
void writeCSCMatrix(const std::string& filename,
                    int n,
                    const std::vector<int>& row_indices,
                    const std::vector<int>& col_ptr,
                    const std::vector<float>& values);

/** @brief 将0-based CSC矩阵转换为0-based COO矩阵 */
void csc_to_coo(int n,
                int nnz,
                const std::vector<int>& col_ptr,
                const std::vector<int>& row_indices,
                const std::vector<float>& values_csc,
                std::vector<int>& rows_coo,
                std::vector<int>& cols_coo,
                std::vector<float>& values_coo);

#endif
