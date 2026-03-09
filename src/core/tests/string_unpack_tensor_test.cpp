// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/string_aligned_buffer.hpp"

using namespace ov;
using namespace testing;

static std::vector<uint8_t> pack_int32s(const std::vector<int32_t>& vals, const std::vector<uint8_t>& tail = {}) {
    std::vector<uint8_t> buf;
    buf.reserve(vals.size() * sizeof(int32_t) + tail.size());
    for (int32_t v : vals) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        buf.insert(buf.end(), p, p + sizeof(int32_t));
    }
    buf.insert(buf.end(), tail.begin(), tail.end());
    return buf;
}

TEST(StringUnpackTensorTest, NegativeNumStringsFails) {
    // num_strings = -1 -> should assert
    auto buf = pack_int32s(std::vector<int32_t>{-1});
    OV_EXPECT_THROW(
        AttributeAdapter<std::shared_ptr<StringAlignedBuffer>>::unpack_string_tensor(reinterpret_cast<const char*>(buf.data()), buf.size()),
        AssertFailure,
        HasSubstr("negative number of strings"));
}

TEST(StringUnpackTensorTest, DecreasingOffsetsFails) {
    // num_strings = 2, pindices: [2, 0, end0=5, end1=3], symbols length small
    std::vector<uint8_t> symbols = {'a', 'b', 'c'};
    auto buf = pack_int32s(std::vector<int32_t>{2, 0, 5, 3}, symbols);
    OV_EXPECT_THROW(
        AttributeAdapter<std::shared_ptr<StringAlignedBuffer>>::unpack_string_tensor(reinterpret_cast<const char*>(buf.data()), buf.size()),
        AssertFailure,
        HasSubstr("begin offset greater than end offset"));
}

TEST(StringUnpackTensorTest, OffsetBeyondBufferFails) {
    // num_strings = 1, pindices: [1, 0, end0=10], but buffer too small
    std::vector<uint8_t> symbols = {'x', 'y'}; // small
    auto buf = pack_int32s(std::vector<int32_t>{1, 0, 10}, symbols);
    OV_EXPECT_THROW(
        AttributeAdapter<std::shared_ptr<StringAlignedBuffer>>::unpack_string_tensor(reinterpret_cast<const char*>(buf.data()), buf.size()),
        AssertFailure,
        HasSubstr("string offset exceeds buffer bounds"));
}
