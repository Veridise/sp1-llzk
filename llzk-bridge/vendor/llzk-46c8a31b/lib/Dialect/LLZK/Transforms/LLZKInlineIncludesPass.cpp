//===-- LLZKInlineIncludesPass.cpp - -llzk-inline-includes pass -*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-inline-includes` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h"
#include "llzk/Dialect/LLZK/Util/IncludeHelper.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_INLINEINCLUDESPASS
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;

namespace {
using IncludeStack = std::vector<std::pair<StringRef, Location>>;

inline bool contains(IncludeStack &stack, StringRef &&loc) {
  auto path_match = [loc](std::pair<StringRef, Location> &p) { return p.first == loc; };
  return std::find_if(stack.begin(), stack.end(), path_match) != stack.end();
}

class InlineIncludesPass : public llzk::impl::InlineIncludesPassBase<InlineIncludesPass> {
  void runOnOperation() override {
    std::vector<std::pair<ModuleOp, IncludeStack>> currLevel = {
        std::make_pair(getOperation(), IncludeStack())
    };
    do {
      std::vector<std::pair<ModuleOp, IncludeStack>> nextLevel = {};
      for (std::pair<ModuleOp, IncludeStack> &curr : currLevel) {
        curr.first.walk([includeStack = std::move(curr.second),
                         &nextLevel](IncludeOp incOp) mutable {
          // Check for cyclic includes
          if (contains(includeStack, incOp.getPath())) {
            auto err = incOp.emitError().append("found cyclic include");
            for (auto it = includeStack.rbegin(); it != includeStack.rend(); ++it) {
              err.attachNote(it->second).append("included from here");
            }
          } else {
            includeStack.push_back(std::make_pair(incOp.getPath(), incOp.getLoc()));
            FailureOr<ModuleOp> result = incOp.inlineAndErase();
            if (succeeded(result)) {
              ModuleOp newMod = std::move(result.value());
              nextLevel.push_back(make_pair(newMod, includeStack));
            }
          }
          // Advance in either case so as many errors as possible are found in a single run.
          return WalkResult::advance();
        });
      }
      currLevel = nextLevel;
    } while (!currLevel.empty());

    markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> llzk::createInlineIncludesPass() {
  return std::make_unique<InlineIncludesPass>();
};
