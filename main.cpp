#include "supernodal_app.hpp"

#ifndef SUPERNODAL_VISUALIZATION_DIR
#define SUPERNODAL_VISUALIZATION_DIR "."
#endif

int main(int argc, char** argv)
{
    SupernodalAppFiles files;
    files.input_filename = "data/A_1215.dat";
    files.ordered_output_filename = "data/A_1215_ordered.dat";
    files.symbolic_cache_filename = "symbolic_analysis.cache";
    files.visualization_directory = SUPERNODAL_VISUALIZATION_DIR;

    return runSupernodalGpuApplication(argc, argv, files);
}
