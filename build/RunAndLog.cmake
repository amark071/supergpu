if(NOT DEFINED EXECUTABLE OR NOT DEFINED WORKING_DIRECTORY OR NOT DEFINED LOG_FILE)
    message(FATAL_ERROR "EXECUTABLE, WORKING_DIRECTORY and LOG_FILE are required")
endif()

string(TIMESTAMP start_time "%Y-%m-%d %H:%M:%S")
string(TIMESTAMP start_epoch "%s")

execute_process(
    COMMAND "${EXECUTABLE}"
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_VARIABLE program_stdout
    ERROR_VARIABLE program_stderr
    RESULT_VARIABLE program_result
)

string(TIMESTAMP end_time "%Y-%m-%d %H:%M:%S")
string(TIMESTAMP end_epoch "%s")
math(EXPR elapsed_seconds "${end_epoch} - ${start_epoch}")

file(APPEND "${LOG_FILE}"
    "[${start_time}] Program started\n"
    "${program_stdout}"
    "${program_stderr}"
    "[${end_time}] Program finished (exit code: ${program_result}, elapsed: ${elapsed_seconds} s)\n\n"
)

message(STATUS "Program output written to: ${LOG_FILE}")

if(NOT program_result EQUAL 0)
    message(FATAL_ERROR "Program exited with code ${program_result}; see ${LOG_FILE}")
endif()
