# GoogleTest
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB nt_test_src "*.cpp")

add_executable(nt_test ${nt_test_src})
target_link_libraries(nt_test GTest::gtest_main)
target_link_libraries(nt_test NetGen)

include(GoogleTest)

# Time in seconds
gtest_discover_tests(nt_test DISCOVERY_TIMEOUT 1000 DISCOVERY_MODE PRE_TEST)
