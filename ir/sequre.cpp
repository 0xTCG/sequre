#include "sequre.h"
#include "expr.h"
// #include "mpc.h"
// #include "mhe.h"
// #include "profiler.h"

namespace sequre {

void Sequre::addIRPasses(codon::ir::transform::PassManager *pm, bool debug) {
  pm->registerPass(std::make_unique<ExpressivenessTransformations>(), debug ? "" : "core-folding-pass-group:2");
  // pm->registerPass(std::make_unique<MPCOptimizations>(), debug ? "" : "sequre-expressiveness-transformation");
  // pm->registerPass(std::make_unique<MHEOptimizations>(), debug ? "" : "sequre-mpc-opt");
  // pm->registerPass(std::make_unique<Profiler>());
}

} // namespace sequre

extern "C" std::unique_ptr<codon::DSL> load() { return std::make_unique<sequre::Sequre>(); }
