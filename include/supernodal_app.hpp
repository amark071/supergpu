#ifndef SUPERNODAL_GPU_INCLUDE_SUPERNODAL_APP_HPP
#define SUPERNODAL_GPU_INCLUDE_SUPERNODAL_APP_HPP

// Application entry point kept separate from main() so the executable driver
// does not contain symbolic analysis, factorization, and verification details.
int runSupernodalGpuApplication(int argc, char** argv);

#endif
