#pragma once

#include "codon/sir/transform/pass.h"
#include "codon/sir/sir.h"

namespace sequre {

class ExpressivenessTransformations : public codon::ir::transform::OperatorPass {
  const std::string KEY = "sequre-expressiveness-transformation";
  std::string getKey() const override { return KEY; }

  void handle(codon::ir::CallInstr *) override;
  
  void transform(codon::ir::CallInstr *);
};

} // namespace sequre
