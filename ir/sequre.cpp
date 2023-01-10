#include "sequre.h"
#include "arithmetics.h"
#include "profiler.h"

namespace sequre {

void Sequre::addIRPasses(codon::ir::transform::PassManager *pm, bool debug) {
  // pm->registerPass(std::make_unique<Profiler>());
  pm->registerPass(std::make_unique<ArithmeticsOptimizations>(), debug ? "" : "core-folding-pass-group:2");
}

} // namespace sequre

extern "C" std::unique_ptr<codon::DSL> load() { return std::make_unique<sequre::Sequre>(); }
