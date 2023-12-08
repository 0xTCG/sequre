#pragma once

#include "codon/cir/cir.h"
#include "codon/cir/transform/pass.h"

namespace sequre {

class Debugger : public codon::ir::transform::OperatorPass {
  const std::string KEY = "sequre-debugger";
  std::string getKey() const override { return KEY; }

  void handle( codon::ir::AssignInstr * ) override;
  
  void attachDebugger( codon::ir::AssignInstr * );
  void replaceVars( codon::ir::Value *, codon::ir::VarValue * );
};

} // namespace sequre
