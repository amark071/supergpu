#ifndef SUPERNODAL_GPU_INCLUDE_SYMBOLIC_CACHE_HPP
#define SUPERNODAL_GPU_INCLUDE_SYMBOLIC_CACHE_HPP

#include "cholmod_symbolic.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

std::uint64_t symbolicStructureHash(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_indices);

bool loadSymbolicCache(
    const std::string& path,
    int n,
    std::size_t input_nonzeros,
    std::uint64_t structure_hash,
    CholmodSymbolicResult& symbolic);

void saveSymbolicCache(
    const std::string& path,
    int n,
    std::size_t input_nonzeros,
    std::uint64_t structure_hash,
    const CholmodSymbolicResult& symbolic);

#endif
