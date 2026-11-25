cmake_minimum_required(VERSION 3.14)
project(Sequre)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wno-return-type-c-linkage")
set(CMAKE_CXX_FLAGS_DEBUG "-g -fno-limit-debug-info")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
include_directories(.)

# (ir/helpers/bet.{h,cpp}, BOSTAN's secure expression tree, are kept for its port and not built)
set(SEQURE_FILES
    ir/analysis/cipher_opt.cpp
    ir/analysis/cipher_opt.h
    ir/helpers/utils.cpp
    ir/helpers/utils.h
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

if(NOT CODON_PATH)
    set(CODON_PATH "$ENV{SEQURE_CODON_PATH}/install")
endif()
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CODON_PATH}/lib/codon/plugins/sequre" CACHE PATH "Use the existing Codon installation" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

target_include_directories(sequre PRIVATE "${CODON_PATH}/include")
target_link_directories(sequre PRIVATE "${CODON_PATH}/lib/codon")
target_link_libraries(sequre PRIVATE codonrt codonc)

install(TARGETS sequre DESTINATION build)
install(FILES plugin.toml DESTINATION .)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/stdlib DESTINATION .)
