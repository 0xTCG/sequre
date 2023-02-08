#pragma once

#include "codon/sir/transform/pass.h"
#include "codon/sir/sir.h"

namespace sequre {

class MPCOptimizations : public codon::ir::transform::OperatorPass {
  const std::string KEY = "sequre-mpc-opt";
  std::string getKey() const override { return KEY; }

  void handle(codon::ir::CallInstr *) override;
};

} // namespace sequre
