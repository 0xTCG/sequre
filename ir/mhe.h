#pragma once

#include "codon/cir/transform/pass.h"
#include "codon/cir/cir.h"

namespace sequre {

class MHEOptimizations : public codon::ir::transform::OperatorPass {
  const std::string KEY = "sequre-mhe-opt";
  std::string getKey() const override { return KEY; }

  void handle(codon::ir::CallInstr *) override;
};

} // namespace sequre
