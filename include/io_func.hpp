#ifndef SUPERNODAL_GPU_INCLUDE_IO_FUNC_HPP
#define SUPERNODAL_GPU_INCLUDE_IO_FUNC_HPP

#include <string>
#include <vector>

/** @brief 读取项目原有的1-based文本CSC文件，转换为0-based内存数据 */
void readCSCMatrix(const std::string& filename,
                   int& n,
                   int& nnz,
                   std::vector<int>& row_indices,
                   std::vector<int>& col_ptr,
                   std::vector<double>& values);

/**
 * @brief 读取二进制CSC文件并转换为求解器使用的int/double CSC。
 *
 * 文件依次存储int8稀疏标志、int32列数、int32行数、8字节nnz、
 * int64列指针、int32行号和double数值。8字节nnz兼容int64以及保存整数的
 * double；读取器通过完整文件长度进行无歧义校验。
 */
void readBinaryCSCMatrix(const std::string& filename,
                         int& n,
                         int& nnz,
                         std::vector<int>& row_indices,
                         std::vector<int>& col_ptr,
                         std::vector<double>& values);

/** @brief 将0-based CSC矩阵写成与A_1215.dat兼容的1-based文本文件 */
void writeCSCMatrix(const std::string& filename,
                    int n,
                    const std::vector<int>& row_indices,
                    const std::vector<int>& col_ptr,
                    const std::vector<double>& values);

#endif
