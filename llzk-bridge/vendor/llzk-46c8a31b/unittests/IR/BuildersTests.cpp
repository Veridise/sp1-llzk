//===-- BuildersTests.cpp - Unit tests for op builders ----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/Builders.h"

#include <gtest/gtest.h>

/* Tests for the ModuleBuilder */

using namespace llzk;

/// TODO: likely a good candidate for property-based testing.
/// A potential good option for a future date: https://github.com/emil-e/rapidcheck
class ModuleBuilderTests : public ::testing::Test {
protected:
  static constexpr auto structAName = "structA";
  static constexpr auto structBName = "structB";
  static constexpr auto structCName = "structC";

  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> mod;
  ModuleBuilder builder;

  ModuleBuilderTests() : context(), mod(createLLZKModule(&context)), builder(mod.get()) {
    context.loadDialect<llzk::LLZKDialect>();
  }

  void SetUp() override {
    // Create a new module and builder for each test.
    mod = createLLZKModule(&context);
    builder = ModuleBuilder(mod.get());
  }
};

TEST_F(ModuleBuilderTests, testModuleOpCreation) { ASSERT_NE(builder.getRootModule(), nullptr); }

TEST_F(ModuleBuilderTests, testStructDefInsertion) {
  builder.insertEmptyStruct(structAName);
  ASSERT_NE(builder.getStruct(structAName), nullptr);
}

TEST_F(ModuleBuilderTests, testFnInsertion) {
  builder.insertFullStruct(structAName);

  auto computeFn = builder.getComputeFn(structAName);
  ASSERT_TRUE(mlir::succeeded(computeFn));
  ASSERT_EQ(computeFn->getBody().getArguments().size(), 0);

  auto constrainFn = builder.getConstrainFn(structAName);
  ASSERT_TRUE(mlir::succeeded(constrainFn));
  ASSERT_EQ(constrainFn->getBody().getArguments().size(), 1);
}

TEST_F(ModuleBuilderTests, testReachabilitySimple) {
  builder.insertComputeOnlyStruct(structAName)
      .insertComputeOnlyStruct(structBName)
      .insertComputeCall(structAName, structBName);

  ASSERT_TRUE(builder.computeReachable(structAName, structBName));
  ASSERT_FALSE(builder.computeReachable(structBName, structAName));
}

TEST_F(ModuleBuilderTests, testReachabilityTransitive) {
  builder.insertComputeOnlyStruct(structAName)
      .insertComputeOnlyStruct(structBName)
      .insertComputeOnlyStruct(structCName)
      .insertComputeCall(structAName, structBName)
      .insertComputeCall(structBName, structCName);

  ASSERT_TRUE(builder.computeReachable(structAName, structBName));
  ASSERT_TRUE(builder.computeReachable(structBName, structCName));
  ASSERT_TRUE(builder.computeReachable(structAName, structCName));
  ASSERT_FALSE(builder.computeReachable(structBName, structAName));
  ASSERT_FALSE(builder.computeReachable(structCName, structAName));
  ASSERT_TRUE(builder.computeReachable(structAName, structAName));
}

TEST_F(ModuleBuilderTests, testReachabilityComputeAndConstrain) {
  builder.insertFullStruct(structAName)
      .insertComputeOnlyStruct(structBName)
      .insertConstrainOnlyStruct(structCName)
      .insertComputeCall(structAName, structBName)
      .insertConstrainCall(structAName, structCName);

  ASSERT_TRUE(builder.computeReachable(structAName, structBName));
  ASSERT_TRUE(builder.constrainReachable(structAName, structCName));
  ASSERT_FALSE(builder.constrainReachable(structAName, structBName));
  ASSERT_FALSE(builder.computeReachable(structAName, structCName));
}

TEST_F(ModuleBuilderTests, testConstruction) {
  builder.insertConstrainOnlyStruct(structAName)
      .insertConstrainOnlyStruct(structBName)
      .insertConstrainOnlyStruct(structCName)
      .insertConstrainCall(structAName, structBName);

  size_t numStructs = 0;
  for (auto s : builder.getRootModule().getOps<llzk::StructDefOp>()) {
    numStructs++;
    size_t numFn = 0;
    for (auto fn : s.getOps<llzk::FuncOp>()) {
      numFn++;
      ASSERT_EQ(fn.getName(), llzk::FUNC_NAME_CONSTRAIN);
    }
    ASSERT_EQ(numFn, 1);
  }
  ASSERT_EQ(numStructs, 3);

  auto aFn = builder.getConstrainFn(structAName);
  ASSERT_TRUE(mlir::succeeded(aFn));
  size_t numOps = 0;
  for ([[maybe_unused]] auto &_ : aFn->getOps()) {
    numOps++;
  }
  ASSERT_EQ(numOps, 2);
}
