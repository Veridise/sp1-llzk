#pragma once

#include <CodegenState.h>

namespace llzk {

struct CodegenStateImpl {
public:
  static CodegenStateImpl &fromWrapper(CodegenState *);
};

} // namespace llzk
