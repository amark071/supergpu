#ifndef SUPERNODAL_GPU_INCLUDE_SUPERNODAL_APP_HPP
#define SUPERNODAL_GPU_INCLUDE_SUPERNODAL_APP_HPP

#include <string>

struct SupernodalAppFiles {
    std::string input_filename;
    std::string ordered_output_filename;
    std::string symbolic_cache_filename;
    std::string visualization_directory;
};

// Application entry point kept separate from main() so the executable driver
// does not contain symbolic analysis, factorization, and verification details.
int runSupernodalGpuApplication(
    int argc,
    char** argv,
    const SupernodalAppFiles& files);

#endif
