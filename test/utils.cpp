#include "utils.h"
#include "shape.h"
#include "gtest/gtest.h"

namespace {

using broadcastShapesParams = std::tuple<nt::Shape, nt::Shape, nt::Shape>;

class UtilsTestBroadcastShapes
    : public testing::TestWithParam<broadcastShapesParams> {};

TEST_P(UtilsTestBroadcastShapes, BroadcastShapes) {
    const auto &[lhs, rhs, expected] = GetParam();
    EXPECT_EQ(nt::broadcastShapes(lhs, rhs), expected);
    EXPECT_EQ(nt::broadcastShapes(rhs, lhs), expected);
}

INSTANTIATE_TEST_SUITE_P(
    BroadcastShapes1, UtilsTestBroadcastShapes,
    ::testing::Combine(::testing::Values(nt::Shape({2, 3, 4})),
                       ::testing::ValuesIn({
                           nt::Shape({2, 3, 4}),
                           nt::Shape({1, 3, 4}),
                           nt::Shape({1, 1, 4}),
                           nt::Shape({1, 1, 1}),
                           nt::Shape({2, 3, 1}),
                           nt::Shape({2, 1, 1}),
                           nt::Shape({2, 1, 4}),
                           nt::Shape({1, 1}),
                           nt::Shape({1}),
                       }),
                       ::testing::Values(nt::Shape({2, 3, 4}))));

TEST(TestRoundUpTo, BasicAssertions) {
    EXPECT_EQ(nt::roundUpTo(10, 1), 10);
    EXPECT_EQ(nt::roundUpTo(10, 2), 10);
    EXPECT_EQ(nt::roundUpTo(10, 4), 12);
    EXPECT_EQ(nt::roundUpTo(10, 7), 14);

    EXPECT_EQ(nt::roundUpTo(0, 3), 3);
    EXPECT_EQ(nt::roundUpTo(1, 3), 3);
}

} // namespace
