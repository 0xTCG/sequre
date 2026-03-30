cmake_minimum_required(VERSION 3.14)
project(Sequre)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -fvisibility-inlines-hidden -Wno-return-type-c-linkage -Wno-gnu-zero-variadic-macro-arguments -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS_DEBUG "-g -fno-limit-debug-info")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
include_directories(.)

set(SEQURE_FILES
    ir/analysis/consecutive_matmul.cpp
    ir/analysis/consecutive_matmul.h
    ir/analysis/dead_code.cpp
    ir/analysis/dead_code.h
    ir/helpers/utils.cpp
    ir/helpers/utils.h
    ir/helpers/bet.cpp
    ir/helpers/bet.h
    ir/obsolete/mpc.cpp
    ir/obsolete/mpc.h
    ir/mhe.cpp
    ir/mhe.h
    ir/expr.cpp
    ir/expr.h
    ir/debugger.cpp
    ir/debugger.h
    ir/sequre.cpp
    ir/sequre.h)
add_library(sequre SHARED ${SEQURE_FILES})

target_include_directories(sequre PRIVATE "${CODON_PATH}/include")
target_link_directories(sequre PRIVATE "${CODON_PATH}/lib/codon")
target_link_libraries(sequre PRIVATE codonrt codonc)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CODON_PATH}/lib/codon" CACHE PATH "Use the existing Codon installation" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

install(TARGETS sequre DESTINATION plugins/sequre/build)
install(FILES plugin.toml DESTINATION plugins/sequre)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/stdlib DESTINATION plugins/sequre)
