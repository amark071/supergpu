#include "supernodal_app.hpp"
#include "unsymmetric_app.hpp"

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

    UnsymmetricAppFiles unsymmetric_files;
    unsymmetric_files.input_filename = "data/A_1215.dat";
    unsymmetric_files.ordered_output_filename =
        "data/A_1215_unsymmetric_ordered.dat";

    const int symmetric_status =
        runSupernodalGpuApplication(argc, argv, files);
    const int unsymmetric_status =
        runUnsymmetricGpuApplication(argc, argv, unsymmetric_files);
    return symmetric_status != 0 ? symmetric_status : unsymmetric_status;
}
