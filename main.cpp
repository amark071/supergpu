#include "supernodal_app.hpp"
#include "unsymmetric_app.hpp"

#ifndef SUPERNODAL_VISUALIZATION_DIR
#define SUPERNODAL_VISUALIZATION_DIR "."
#endif

#include <exception>
#include <iostream>
#include <string>

namespace {

enum class InputMatrixFormat {
    TextCsc,
    BinaryCsc
};

bool hasArgument(int argc, char** argv, const char* option)
{
    for (int argument = 1; argument < argc; ++argument) {
        if (std::string(argv[argument]) == option) {
            return true;
        }
    }
    return false;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const InputMatrixFormat input_format = InputMatrixFormat::BinaryCsc;
        const bool input_is_binary_csc =
            input_format == InputMatrixFormat::BinaryCsc;
        const std::string input_filename =
            "data/150w/stiffness_matrix.bin";
        SupernodalAppFiles files;
        files.input_filename = input_filename;
        files.ordered_output_filename = "data/bin_ordered.dat";
        files.symbolic_cache_filename = "symbolic_analysis.cache";
        files.visualization_directory = SUPERNODAL_VISUALIZATION_DIR;
        files.input_is_binary_csc = input_is_binary_csc;

        UnsymmetricAppFiles unsymmetric_files;
        unsymmetric_files.input_filename = input_filename;
        unsymmetric_files.ordered_output_filename =
            "data/A_1215_unsymmetric_ordered.dat";
        unsymmetric_files.matching_enabled =
            !hasArgument(argc, argv, "--disable-matching");
        unsymmetric_files.input_is_binary_csc = input_is_binary_csc;

        const int symmetric_status =
            runSupernodalGpuApplication(argc, argv, files);
        const int unsymmetric_status =
            runUnsymmetricGpuApplication(argc, argv, unsymmetric_files);
        return symmetric_status != 0 ? symmetric_status : unsymmetric_status;
    } catch (const std::exception& error) {
        std::cerr << "Application failed: " << error.what() << '\n';
        return 10;
    }
}
