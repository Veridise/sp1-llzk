//===-- SymbolLookup.h - Symbol Lookup Functions ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines methods symbol lookup across LLZK operations and
/// included files.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>

#include <vector>

namespace llzk {

constexpr char LANG_ATTR_NAME[] = "veridise.lang";

using ManagedResources =
    std::shared_ptr<std::pair<mlir::OwningOpRef<mlir::ModuleOp>, mlir::SymbolTableCollection>>;

class SymbolLookupResultUntyped {
public:
  SymbolLookupResultUntyped() : op(nullptr) {}
  SymbolLookupResultUntyped(mlir::Operation *op) : op(op) {}

  /// Access the internal operation.
  mlir::Operation *operator->();
  mlir::Operation &operator*();
  mlir::Operation &operator*() const;
  mlir::Operation *get();
  mlir::Operation *get() const;

  /// True iff the symbol was found.
  operator bool() const;

  std::vector<llvm::StringRef> getIncludeSymNames() { return includeSymNameStack; }

  mlir::SymbolTableCollection *getSymbolTableCache() {
    if (managedResources) {
      return &managedResources->second;
    } else {
      return nullptr;
    }
  }

  /// Adds a pointer to the set of resources the result has to manage the lifetime of.
  void manage(mlir::OwningOpRef<mlir::ModuleOp> &&ptr, mlir::SymbolTableCollection &&tables);

  /// Adds the symbol name from the IncludeOp that caused the module to be loaded.
  void trackIncludeAsName(llvm::StringRef includeOpSymName);

  bool operator==(const SymbolLookupResultUntyped &rhs) const { return op == rhs.op; }

private:
  mlir::Operation *op;
  /// Owns the ModuleOp that contains 'op' if it was loaded via an IncludeOp along with the
  /// SymbolTableCollection for that ModuleOp which should be used for lookups involving 'op'.
  ManagedResources managedResources;
  /// Stack of symbol names from the IncludeOp that were traversed in order to load the Operation.
  std::vector<llvm::StringRef> includeSymNameStack;

  friend class Within;
};

template <typename T> class SymbolLookupResult {
public:
  SymbolLookupResult(SymbolLookupResultUntyped &&inner) : inner(std::move(inner)) {}

  /// Access the internal operation as type T.
  /// Follows the behaviors of llvm::dyn_cast if the internal operation cannot cast to that type.
  T operator->() { return llvm::dyn_cast<T>(*inner); }
  T operator*() { return llvm::dyn_cast<T>(*inner); }
  const T operator*() const { return llvm::dyn_cast<T>(*inner); }
  T get() { return llvm::dyn_cast<T>(inner.get()); }
  T get() const { return llvm::dyn_cast<T>(inner.get()); }

  operator bool() const { return inner && llvm::isa<T>(*inner); }

  std::vector<llvm::StringRef> getIncludeSymNames() { return inner.getIncludeSymNames(); }

  bool operator==(const SymbolLookupResult<T> &rhs) const { return inner == rhs.inner; }

private:
  SymbolLookupResultUntyped inner;

  friend class Within;
};

class Within {
public:
  /// Lookup within the top-level (root) module
  Within() : from(nullptr) {}
  /// Lookup within the given Operation (cannot be nullptr)
  Within(mlir::Operation *op) : from(op) { assert(op && "cannot lookup within nullptr"); }
  /// Lookup within the Operation of the given result and transfer managed resources
  Within(SymbolLookupResultUntyped &&res) : from(std::move(res)) {}
  /// Lookup within the Operation of the given result and transfer managed resources
  template <typename T> Within(SymbolLookupResult<T> &&res) : Within(std::move(res.inner)) {}

  Within(const Within &) = delete;
  Within(Within &&other) : from(std::move(other.from)) {}
  Within &operator=(const Within &) = delete;
  Within &operator=(Within &&);

  inline static Within root() { return Within(); }

  mlir::FailureOr<SymbolLookupResultUntyped> lookup(
      mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr symbol, mlir::Operation *origin,
      bool reportMissing = true
  ) &&;

private:
  std::variant<mlir::Operation *, SymbolLookupResultUntyped> from;
};

inline mlir::FailureOr<SymbolLookupResultUntyped> lookupSymbolIn(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr symbol, Within &&lookupWithin,
    mlir::Operation *origin, bool reportMissing = true
) {
  return std::move(lookupWithin).lookup(tables, symbol, origin, reportMissing);
}

inline mlir::FailureOr<SymbolLookupResultUntyped> lookupTopLevelSymbol(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr symbol, mlir::Operation *origin,
    bool reportMissing = true
) {
  return Within().lookup(tables, symbol, origin, reportMissing);
}

template <typename T>
inline mlir::FailureOr<SymbolLookupResult<T>> lookupSymbolIn(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr symbol, Within &&lookupWithin,
    mlir::Operation *origin
) {
  auto found = lookupSymbolIn(tables, symbol, std::move(lookupWithin), origin);
  if (mlir::failed(found)) {
    return mlir::failure(); // lookupSymbolIn() already emits a sufficient error message
  }
  // Keep a copy of the op ptr in case we need it for displaying diagnostics
  mlir::Operation *op = found->get();
  // ... since the untyped result gets moved here into a typed result.
  SymbolLookupResult<T> ret(std::move(*found));
  if (!ret) {
    return origin->emitError() << "symbol \"" << symbol << "\" references a '" << op->getName()
                               << "' but expected a '" << T::getOperationName() << "'";
  }
  return ret;
}

template <typename T>
inline mlir::FailureOr<SymbolLookupResult<T>> lookupTopLevelSymbol(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr symbol, mlir::Operation *origin
) {
  return lookupSymbolIn<T>(tables, symbol, Within(), origin);
}

} // namespace llzk
