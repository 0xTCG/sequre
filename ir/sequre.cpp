#include "sequre.h"
#include "expr.h"
#include "obsolete/mpc.h"
#include "mhe.h"
#include "debugger.h"

namespace sequre {

void Sequre::addIRPasses( codon::ir::transform::PassManager *pm, bool debug ) {
  // NOTE: ExpressivenessTransformations (and the MHEOptimizations pass chained
  // after it below) realize brand-new function subtrees when they rewrite secure
  // operators (e.g. `ndarray @ Ciphertensor` -> `secure_matmul(mpc, ...)`). Those
  // subtrees can contain `@par` loops (CKKS/ring code). Such loops must be lowered
  // by `core-parallel-openmp`, otherwise the LLVM visitor aborts with
  // "parallel for-loop not lowered". We must therefore run BEFORE that pass in
  // every pipeline. The old placement was buggy in both modes: debug/JIT has no
  // folding pass groups so "" appended at the end (after openmp), and in release
  // `core-folding-pass-group:2` is itself a folding group that runs after openmp
  // lowering -- so both left freshly realized `@par` loops unlowered.
  // `core-parallel-openmp` is present in every pipeline, so anchor to it.
  pm->registerPass(std::make_unique<ExpressivenessTransformations>(), "core-parallel-openmp");
  pm->registerPass(std::make_unique<MHEOptimizations>(), "sequre-expressiveness-transformation");
}

} // namespace sequre

extern "C" std::unique_ptr<codon::DSL> load() { return std::make_unique<sequre::Sequre>(); }
