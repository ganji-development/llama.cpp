# The transcriber's guts, built once and used by both the CLI and the shared
# library so the two cannot drift. An OBJECT library rather than a static one
# because the sources have to be position-independent for the DLL.
add_library(iaamt-core OBJECT
    iaamt-model.cpp
    iaamt-cqt.cpp
    iaamt-graph.cpp
    iaamt-decode.cpp
    iaamt-velocity.cpp
    iaamt-beatchord.cpp
    iaamt-midi.cpp
)

target_link_libraries(iaamt-core PUBLIC ggml ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(iaamt-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/vendor
)

target_compile_features(iaamt-core PUBLIC cxx_std_17)
set_target_properties(iaamt-core PROPERTIES POSITION_INDEPENDENT_CODE ON)

#
# llama-iaamt-cli: the standalone tool.
#
set(TARGET llama-iaamt-cli)

add_executable(${TARGET} iaamt-cli.cpp)

# Deliberately links ggml only: this tool has no vocabulary, no KV cache and no
# HTTP dependency, so it does not need llama or common.
target_link_libraries(${TARGET} PRIVATE iaamt-core)

install(TARGETS ${TARGET} RUNTIME)

#
# iaamt: the same transcriber behind a flat C ABI.
#
# Exists because iaamt.h is a C++ header - std::vector and std::string across
# its signatures - so a consumer must share this tree's compiler, standard
# library and CRT. MIDIGen does not: its core builds with MSVC against the
# static runtime while this builds with ROCm's clang against the dynamic one.
# The DLL lets it resolve iaamt-c.h's symbols by name at run time instead,
# which is what libs/model-core already does for llama.dll.
add_library(iaamt SHARED iaamt-c.cpp)

target_link_libraries(iaamt PRIVATE iaamt-core)

# Only the C ABI is exported; everything else stays internal to the DLL.
target_compile_definitions(iaamt PRIVATE IAAMT_C_BUILD)

set_target_properties(iaamt PROPERTIES
    PUBLIC_HEADER iaamt-c.h
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
)

install(TARGETS iaamt
    RUNTIME
    LIBRARY
    PUBLIC_HEADER
)
