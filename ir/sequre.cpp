#include "sequre.h"
#include "expr.h"
#include "obsolete/mpc.h"
#include "mhe.h"
#include "debugger.h"

namespace sequre {

void Sequre::addIRPasses( codon::ir::transform::PassManager *pm, bool debug ) {
  pm->registerPass(std::make_unique<ExpressivenessTransformations>(), debug ? "" : "core-folding-pass-group:2");
  pm->registerPass(std::make_unique<MPCOptimizations>(), "sequre-expressiveness-transformation");
  pm->registerPass(std::make_unique<MHEOptimizations>(), "sequre-mpc-opt");
  pm->registerPass(std::make_unique<Debugger>(), "sequre-mhe-opt");
}

} // namespace sequre

extern "C" std::unique_ptr<codon::DSL> load() { return std::make_unique<sequre::Sequre>(); }
