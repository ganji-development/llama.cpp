set(TARGET llama-iaamt-cli)

add_executable(${TARGET}
    iaamt-cli.cpp
    iaamt-model.cpp
    iaamt-cqt.cpp
    iaamt-graph.cpp
    iaamt-decode.cpp
    iaamt-velocity.cpp
    iaamt-beatchord.cpp
    iaamt-midi.cpp
)

# Deliberately links ggml only: this tool has no vocabulary, no KV cache and no
# HTTP dependency, so it does not need llama or common.
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/vendor
)

target_compile_features(${TARGET} PRIVATE cxx_std_17)

install(TARGETS ${TARGET} RUNTIME)
