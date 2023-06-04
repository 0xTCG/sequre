#pragma once

#include "codon/sir/transform/pass.h"
#include "codon/sir/sir.h"

namespace sequre {

class Profiler : public codon::ir::transform::OperatorPass {
  const std::string KEY = "sequre-profiler";
  std::string getKey() const override { return KEY; }

  void handle(codon::ir::CallInstr *) override;

  void attachProfiler(codon::ir::CallInstr *);
};

} // namespace sequre
