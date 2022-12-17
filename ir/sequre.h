#pragma once

#include "codon/dsl/dsl.h"

namespace sequre {

class Sequre : public codon::DSL {
public:
  void addIRPasses(codon::ir::transform::PassManager *pm, bool debug) override;
};

} // namespace sequre
