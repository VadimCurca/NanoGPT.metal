#include "tensor.h"
#include "shape.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace {

// NOLINTBEGIN
TEST(TensorTestDataAccess, BasicAssertions) {
    float a[2][3][3] = {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                        {{10, 20, 30}, {40, 50, 60}, {70, 80, 90}}};

    nt::Tensor t(nt::Shape({2, 3, 3}), **a);

    float output1[2][3][3];
    for (size_t k = 0; k < t.getShape()[0]; k++) {
        for (size_t i = 0; i < t.getShape()[1]; i++) {
            for (size_t j = 0; j < t.getShape()[2]; j++) {
                output1[k][i][j] = t({k, i, j});
            }
        }
    }
    EXPECT_EQ(0, std::memcmp(a, output1, sizeof(a)));

    t({0, 1, 2}) = 42.0F;
    a[0][1][2] = 42.0F;

    float output2[2][3][3];
    for (size_t k = 0; k < t.getShape()[0]; k++) {
        for (size_t i = 0; i < t.getShape()[1]; i++) {
            for (size_t j = 0; j < t.getShape()[2]; j++) {
                auto tI = t({k, i, j});
                output2[k][i][j] = tI;
                std::cout << output2[k][i][j] << ' ';
            }
        }
    }
    EXPECT_EQ(0, std::memcmp(a, output2, sizeof(a)));

    EXPECT_NO_THROW(t({0, 1, 2}));

    EXPECT_THROW(t({0, 1}), std::runtime_error);
    EXPECT_THROW(t({0, 1, 2, 3}), std::runtime_error);
}

TEST(TensorTestDefaultConstructor, BasicAssertions) {
    nt::Tensor t;
    EXPECT_EQ(t.begin(), nullptr);
    EXPECT_EQ(t.getShape(), nt::Shape({}));
}

TEST(TensorTestAllocateUnitializedData, BasicAssertions) {
    nt::Tensor t(nt::Shape({2, 3, 4}));
    EXPECT_NE(t.begin(), nullptr);
    EXPECT_EQ(t.getShape(), nt::Shape({2, 3, 4}));

    t({1, 2, 3}) = 2;
    EXPECT_EQ(t({1, 2, 3}), 2);
}

TEST(TensorTestAllocateAndInitializeWithConstant, BasicAssertions) {
    const unsigned n = 10;
    const unsigned m = 20;
    const float constant = 3.14;
    nt::Tensor t(nt::Shape({n, m}), constant);

    EXPECT_TRUE(
        std::all_of(t.begin(), t.end(), [&](auto x) { return x == constant; }));
}

TEST(TensorTestAllocateAndCopyScalar, BasicAssertions) {
    float x = 3.14;
    nt::Tensor t(3.14);

    EXPECT_EQ(t.getShape(), nt::Shape({1}));
    EXPECT_NE(t.begin(), nullptr);
    EXPECT_EQ(t({0}), x);
}

TEST(TensorTestStrides, BasicAssertions) {
    nt::Tensor a(nt::Shape({1, 3, 224, 224}));
    std::vector<size_t> expected({150528, 50176, 224, 1});

    auto strides = a.getStrides();
    EXPECT_EQ(strides, expected);
}

TEST(TensorTestReshape, BasicAssertions) {
    nt::Tensor a(nt::Shape({1, 3, 224, 224}));
    nt::Shape shape;

    shape = {1, 1, 224, 672};
    a.reshape(shape);
    EXPECT_EQ(a.getShape(), shape);

    shape = {672, 224};
    a.reshape(shape);
    EXPECT_EQ(a.getShape(), shape);

    shape = {672, 224, 1, 1};
    a.reshape(shape);
    EXPECT_EQ(a.getShape(), shape);

    shape = {150528};
    a.reshape(shape);
    EXPECT_EQ(a.getShape(), shape);

    shape = {1, 150528 + 1, 1, 1};
    EXPECT_THROW(a.reshape(shape);, std::runtime_error);

    shape = {1};
    EXPECT_THROW(a.reshape(shape);, std::runtime_error);

    shape = {1, 0, 3, 224, 224};
    EXPECT_THROW(a.reshape(shape);, std::runtime_error);
}

TEST(TensorTestReshapeInference, BasicAssertions) {
    nt::Shape shape(nt::Shape({1, 3, 224, 224}));
    nt::Tensor a(shape);
    std::vector<int64_t> newShape;

    newShape = {-1, 3, 224, 224};
    a.reshape(newShape);
    EXPECT_EQ(a.getShape(), shape);

    newShape = {1, -1, 224, 224};
    a.reshape(newShape);
    EXPECT_EQ(a.getShape(), shape);

    newShape = {1, 3, -1, 224};
    a.reshape(newShape);
    EXPECT_EQ(a.getShape(), shape);

    newShape = {1, 3, 224, -1};
    a.reshape(newShape);
    EXPECT_EQ(a.getShape(), shape);

    newShape = {-1, 224, 224};
    a.reshape(newShape);
    EXPECT_EQ(a.getShape(), nt::Shape({3, 224, 224}));

    newShape = {3, -1};
    a.reshape(newShape);
    EXPECT_EQ(a.getShape(), nt::Shape({3, 224 * 224}));

    newShape = {-1};
    a.reshape(newShape);
    EXPECT_EQ(a.getShape(), nt::Shape({3 * 224 * 224}));

    newShape = {-1, -1, 224, 224};
    EXPECT_THROW(a.reshape(newShape), std::runtime_error);

    newShape = {1, -2, 224, 224};
    EXPECT_THROW(a.reshape(newShape), std::runtime_error);
}

TEST(TensorTestToString, BasicAssertions) {
    nt::Tensor a(nt::Shape{2, 3, 4});
    std::iota(a.begin(), a.end(), 1);
    std::cout << a << '\n';
}

// NOLINTEND

} // namespace
