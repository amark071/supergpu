#ifndef SUPERNODAL_GPU_INCLUDE_UNSYMMETRIC_APP_HPP
#define SUPERNODAL_GPU_INCLUDE_UNSYMMETRIC_APP_HPP

#include <string>

struct UnsymmetricAppFiles {
    std::string input_filename;
    std::string ordered_output_filename;
};

int runUnsymmetricGpuApplication(
    int argc,
    char** argv,
    const UnsymmetricAppFiles& files);

#endif
