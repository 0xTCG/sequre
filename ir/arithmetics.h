#pragma once

#include "codon/sir/transform/pass.h"
#include "codon/sir/sir.h"

namespace sequre {

class ArithmeticsOptimizations : public codon::ir::transform::OperatorPass {
  const std::string KEY = "sequre-arithmetic-opt";
  std::string getKey() const override { return KEY; }

  void handle(codon::ir::CallInstr *) override;

  void applyBeaverOptimizations(codon::ir::CallInstr *);
  void applyPolynomialOptimizations(codon::ir::CallInstr *);
  void applyFactorizationOptimizations(codon::ir::CallInstr *);
  void applyOptimizations(codon::ir::CallInstr *);
};

} // namespace sequre
