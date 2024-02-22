cmake_minimum_required(VERSION 2.8.3)

project(HelloWorld)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

aux_source_directory(./src SRC)

include_directories(./inc)

add_executable(hello ${SRC})

# set up the test
enable_testing()

macro(test TESTNAME)
    
    ADD_TEST(NAME test_hello_${TESTNAME} COMMAND hello ${TESTNAME})
    SET_TESTS_PROPERTIES(test_hello_${TESTNAME} 
        PROPERTIES PASS_REGULAR_EXPRESSION "Hello, ${TESTNAME}")

endmacro()

test(KCN)
test(CODE)