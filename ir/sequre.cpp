#include "sequre.h"
#include "mpc.h"
#include "profiler.h"

namespace sequre {

void Sequre::addIRPasses(codon::ir::transform::PassManager *pm, bool debug) {
  // pm->registerPass(std::make_unique<Profiler>());
  pm->registerPass(std::make_unique<MPCOptimizations>(), debug ? "" : "core-folding-pass-group:2");
  pm->registerPass(std::make_unique<MHEOptimizations>(), debug ? "" : "sequre-mpc-opt");
}

} // namespace sequre

extern "C" std::unique_ptr<codon::DSL> load() { return std::make_unique<sequre::Sequre>(); }
