//===-- TypesTests.cpp - Unit tests for LLZK types --------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Types.h"

#include <gtest/gtest.h>

using namespace llzk;
using namespace mlir;

class TypeTests : public ::testing::Test {
public:
  MLIRContext ctx;

protected:
  TypeTests() : ctx() { ctx.loadDialect<llzk::LLZKDialect>(); }
};

TEST_F(TypeTests, testArrayTypeCloneSuccessNewType) {
  IntegerType tyBool = IntegerType::get(&ctx, 1);
  IndexType tyIndex = IndexType::get(&ctx);
  ArrayType a = ArrayType::get(tyIndex, {2, 2});
  ArrayType b = a.cloneWith(std::nullopt, tyBool);
  ASSERT_EQ(b.getElementType(), tyBool);
  ASSERT_EQ(b.getShape(), ArrayRef(std::vector<int64_t>({2, 2})));
}

TEST_F(TypeTests, testArrayTypeCloneSuccessNewShape) {
  IndexType tyIndex = IndexType::get(&ctx);
  ArrayType a = ArrayType::get(tyIndex, {2, 2});
  std::vector<int64_t> newShapeVec({2, 3, 2});
  ArrayRef newShape(newShapeVec);
  ArrayType b = a.cloneWith(std::make_optional(newShape), tyIndex);
  ASSERT_EQ(b.getElementType(), tyIndex);
  ASSERT_EQ(b.getShape(), newShape);
}

TEST_F(TypeTests, testArrayTypeCloneWithEmptyShapeError) {
  EXPECT_DEATH(
      {
        IndexType tyIndex = IndexType::get(&ctx);
        ArrayType a = ArrayType::get(tyIndex, {2, 2});
        std::vector<int64_t> newShapeVec;
        ArrayRef newShape(newShapeVec);
        a.cloneWith(std::make_optional(newShape), tyIndex);
      },
      "error: array must have at least one dimension"
  );
}

TEST_F(TypeTests, testArrayTypeGetWithAttributeEmptyShapeError) {
  EXPECT_DEATH(
      {
        IndexType tyIndex = IndexType::get(&ctx);
        std::vector<Attribute> newDimsVec;
        ArrayRef<Attribute> dimensionSizes(newDimsVec);
        ArrayType::get(tyIndex, dimensionSizes);
      },
      "error: array must have at least one dimension"
  );
}

TEST_F(TypeTests, testArrayTypeGetWithAttributeWrongAttrKindError) {
  EXPECT_DEATH(
      {
        IndexType tyIndex = IndexType::get(&ctx);
        std::vector<Attribute> newDimsVec = {UnitAttr::get(&ctx)};
        ArrayRef<Attribute> dimensionSizes(newDimsVec);
        ArrayType::get(tyIndex, dimensionSizes);
      },
      "error: Array dimension must be one of .* but found 'builtin.unit'"
  );
}

TEST_F(TypeTests, testStructTypeIsConcreteNoParams) {
  StructType t = StructType::get(FlatSymbolRefAttr::get(&ctx, "TheName"));
  // When the StructType has no parameters, isConcreteType() passes regardless of
  // `allowStructParams` flag.
  ASSERT_TRUE(isConcreteType(t, true));
  ASSERT_TRUE(isConcreteType(t, false));
}

TEST_F(TypeTests, testStructTypeIsConcreteWithParams) {
  Attribute a1 = IntegerAttr::get(IndexType::get(&ctx), 128);
  Attribute a2 = IntegerAttr::get(IndexType::get(&ctx), 15);
  StructType t = StructType::get(FlatSymbolRefAttr::get(&ctx, "TheName"), ArrayRef {a1, a2});
  // When the StructType has parameters, isConcreteType() passes when `allowStructParams` flag is
  // `true` but fails when it is `false`.
  ASSERT_TRUE(isConcreteType(t, true));
  ASSERT_FALSE(isConcreteType(t, false));
}

TEST_F(TypeTests, testShortString) {
  OpBuilder bldr(&ctx);
  EXPECT_EQ("b", shortString(bldr.getIntegerType(1)));
  EXPECT_EQ("i", shortString(bldr.getIndexType()));
  EXPECT_EQ("!t<@A>", shortString(TypeVarType::get(FlatSymbolRefAttr::get(&ctx, "A"))));
  EXPECT_EQ(
      "!a<b:4_235_123>",
      shortString(ArrayType::get(bldr.getIntegerType(1), ArrayRef<int64_t> {4, 235, 123}))
  );
  EXPECT_EQ("!s<@S1>", shortString(StructType::get(FlatSymbolRefAttr::get(&ctx, "S1"))));
  EXPECT_EQ(
      "!s<@S1_43>",
      shortString(StructType::get(
          FlatSymbolRefAttr::get(&ctx, "S1"),
          ArrayAttr::get(&ctx, ArrayRef<Attribute> {bldr.getIntegerAttr(bldr.getIndexType(), 43)})
      ))
  );
  {
    auto innerStruct = StructType::get(
        FlatSymbolRefAttr::get(&ctx, "S1"),
        ArrayAttr::get(&ctx, ArrayRef<Attribute> {bldr.getIntegerAttr(bldr.getIndexType(), 43)})
    );
    auto params = ArrayAttr::get(
        &ctx,
        ArrayRef<Attribute> {
            bldr.getIntegerAttr(bldr.getIndexType(), 43), FlatSymbolRefAttr::get(&ctx, "ParamName"),
            TypeAttr::get(ArrayType::get(FeltType::get(&ctx), ArrayRef<int64_t> {3, 5, 1, 5, 7})),
            TypeAttr::get(innerStruct), AffineMapAttr::get(bldr.getDimIdentityMap())
        }
    );
    EXPECT_EQ(
        "!s<@Top_43_@ParamName_!a<f:3_5_1_5_7>_!s<@S1_43>_!m<(d0)->(d0)>>",
        shortString(StructType::get(FlatSymbolRefAttr::get(&ctx, "Top"), params))
    );
  }

  // No protection/escaping of special characters in the original name
  EXPECT_EQ("!s<@S1_!a<>>", shortString(StructType::get(FlatSymbolRefAttr::get(&ctx, "S1_!a<>"))));

  // Empty string produces "?"
  EXPECT_EQ("?", shortString(FlatSymbolRefAttr::get(&ctx, "")));
  EXPECT_EQ("?", shortString(FlatSymbolRefAttr::get(&ctx, StringRef())));

  {
    constexpr char withNull[] = {'a', 'b', '\0', 'c', 'd'};
    EXPECT_EQ(
        "5_@head_@ab_@Good_2",
        // clang-format off
        shortString(ArrayAttr::get( &ctx, ArrayRef<Attribute> {
          bldr.getIntegerAttr(bldr.getIndexType(), 5),
          FlatSymbolRefAttr::get(&ctx, "head\0_tail"),
          FlatSymbolRefAttr::get(&ctx, withNull),
          FlatSymbolRefAttr::get(&ctx, "Good"),
          bldr.getIntegerAttr(bldr.getIndexType(), 2)
        }))
        // clang-format on
    );
  }
}
