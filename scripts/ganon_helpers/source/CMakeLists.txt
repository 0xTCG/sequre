cmake_minimum_required (VERSION 3.4)
project (ibf_extractor CXX)
 
# add seqan3 to search path
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../seqan3/build_system")
 
# require seqan3 with a version between >=3.0.0 and <4.0.0
find_package (seqan3 3.0 REQUIRED)
 
# build app with seqan3
add_executable (ibf_extractor ibf_extractor.cpp)
target_link_libraries (ibf_extractor seqan3::seqan3)

add_executable (test test.cpp)
target_link_libraries (test seqan3::seqan3)
