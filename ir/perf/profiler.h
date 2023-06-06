#pragma once

#include "codon/cir/transform/pass.h"
#include "codon/cir/cir.h"

namespace sequre {

class Profiler : public codon::ir::transform::OperatorPass {
  const std::string KEY = "sequre-profiler";
  std::string getKey() const override { return KEY; }

  void handle(codon::ir::CallInstr *) override;

  void attachProfiler(codon::ir::CallInstr *);
};

} // namespace sequre
