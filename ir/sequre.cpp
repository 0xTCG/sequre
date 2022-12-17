#include "sequre.h"
#include "arithmetics.h"

namespace sequre {

void Sequre::addIRPasses(codon::ir::transform::PassManager *pm, bool debug) {
  pm->registerPass(std::make_unique<ArithmeticsOptimizations>(), "core-folding-pass-group:2");
}

} // namespace sequre

extern "C" std::unique_ptr<codon::DSL> load() { return std::make_unique<sequre::Sequre>(); }
