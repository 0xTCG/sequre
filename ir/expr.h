#pragma once

#include "codon/cir/cir.h"
#include "codon/cir/transform/pass.h"

namespace sequre {

class ExpressivenessTransformations : public codon::ir::transform::OperatorPass {
  const std::string KEY = "sequre-expressiveness-transformation";
  std::string getKey() const override { return KEY; }

  void handle( codon::ir::CallInstr * ) override;
  
  void enableSecurity( codon::ir::CallInstr * );
};

} // namespace sequre
