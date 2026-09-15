#include "unsymmetric_app.hpp"

#include <exception>
#include <iostream>
#include <string>

int runDoubleSymmetricSolver(
    const std::string& filename,
    bool input_is_binary_csc);

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
        //const InputMatrixFormat input_format = InputMatrixFormat::BinaryCsc;
        const InputMatrixFormat input_format = InputMatrixFormat::TextCsc;
        const bool input_is_binary_csc =
            input_format == InputMatrixFormat::BinaryCsc;
        const std::string input_filename =
            "data/A_63666.dat";
        if (hasArgument(argc, argv, "--double-symmetric-only")) {
            return runDoubleSymmetricSolver(
                input_filename, input_is_binary_csc);
        }
        UnsymmetricAppFiles unsymmetric_files;
        unsymmetric_files.input_filename = input_filename;
        unsymmetric_files.ordered_output_filename =
            "data/oA_63666.dat";
        unsymmetric_files.matching_enabled =
            !hasArgument(argc, argv, "--disable-matching");
        unsymmetric_files.input_is_binary_csc = input_is_binary_csc;

        const int symmetric_status = runDoubleSymmetricSolver(
            input_filename, input_is_binary_csc);
        const int unsymmetric_status =
            runUnsymmetricGpuApplication(argc, argv, unsymmetric_files);
        return symmetric_status != 0 ? symmetric_status : unsymmetric_status;
    } catch (const std::exception& error) {
        std::cerr << "Application failed: " << error.what() << '\n';
        return 10;
    }
}
