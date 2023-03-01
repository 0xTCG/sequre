#pragma once

#include "codon/sir/transform/pass.h"
#include "codon/sir/sir.h"

namespace sequre {

class MHEOptimizations : public codon::ir::transform::OperatorPass {
  const std::string KEY = "sequre-mhe-opt";
  std::string getKey() const override { return KEY; }

  void handle(codon::ir::CallInstr *) override;
};

} // namespace sequre
