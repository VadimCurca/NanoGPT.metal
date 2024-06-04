#include "shape.h"
#include "gtest/gtest.h"
#include <_types/_uint32_t.h>
#include <algorithm>
#include <numeric>
#include <vector>

namespace {

TEST(ShapeTestDefaultConstructor, BasicAssertions) {
    nt::Shape shape;
    EXPECT_EQ(static_cast<std::vector<nt::Shape::value_type>>(shape),
              std::vector<nt::Shape::value_type>());
    EXPECT_EQ(shape.dim(), 0);
    EXPECT_EQ(shape.numel(), 0);
}

TEST(ShapeTestConstructFromInitializerList, BasicAssertions) {
    const nt::Shape shape({3, 224, 512});

    EXPECT_EQ(static_cast<std::vector<nt::Shape::value_type>>(shape),
              std::vector<nt::Shape::value_type>({3, 224, 512}));
    EXPECT_EQ(shape.dim(), 3);
    EXPECT_EQ(shape.numel(), 3 * 224 * 512);
}

TEST(ShapeTestConstructFromStdVector, BasicAssertions) {
    const nt::Shape shape(
        nt::Shape::fromVector(std::vector<nt::Shape::value_type>{3, 224, 512}));

    EXPECT_EQ(static_cast<std::vector<nt::Shape::value_type>>(shape),
              std::vector<nt::Shape::value_type>({3, 224, 512}));
    EXPECT_EQ(shape.dim(), 3);
    EXPECT_EQ(shape.numel(), 3 * 224 * 512);
}

TEST(ShapeTestAccesors, BasicAssertions) {
    nt::Shape shape({1, 3, 224, 224}); // NOLINT
    const nt::Shape expected({32, 16, 4, 2});

    for (size_t i = 0; i < shape.dim(); i++) {
        shape[i] = expected[i];
    }

    EXPECT_EQ(shape, expected);

    shape = {1, 3, 224, 224}; // NOLINT

    std::for_each(shape.begin(), shape.end(),
                  [](nt::Shape::value_type &x) { x++; });

    EXPECT_EQ(shape, nt::Shape({2, 4, 225, 225}));

    std::iota(shape.rbegin(), shape.rend(), 1);

    EXPECT_EQ(shape, nt::Shape({4, 3, 2, 1}));
}

} // namespace
