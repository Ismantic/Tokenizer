set(LIB_SRCS
    ustr.cc
    sentence.cc
    sentencepiece_counter.cc
    sentencepiece_tokenizer.cc
    normalizer.cc
    bytepiece_tokenizer.cc
    bytepiece_counter.cc
    piece_tokenizer.cc
    piece_counter.cc
    naive_tokenizer.cc
    naive_counter.cc
)

add_library(isma_tokenizer_lib ${LIB_SRCS})
target_include_directories(isma_tokenizer_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(isma-tokenizer main.cc)
target_link_libraries(isma-tokenizer PRIVATE isma_tokenizer_lib)

add_executable(isma_tokenizer_test tokenizer_test.cc test.cc ustr_test.cc)
target_link_libraries(isma_tokenizer_test PRIVATE isma_tokenizer_lib)
