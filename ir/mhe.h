#pragma once

#include "codon/cir/transform/pass.h"
#include "codon/cir/cir.h"

namespace sequre {

// @mhe_cipher_opt: each decorated (realized) function is rewritten once, however many call sites it has.
// @mhe_enc_opt is accepted and does nothing.
class MHEOptimizations : public codon::ir::transform::Pass {
  const std::string KEY = "sequre-mhe-opt";

public:
  std::string getKey() const override { return KEY; }
  void run( codon::ir::Module * ) override;
};

} // namespace sequre
