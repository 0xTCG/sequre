#include "sequre.h"
#include "expr.h"
#include "obsolete/mpc.h"
#include "mhe.h"
#include "debugger.h"

namespace sequre {

void Sequre::addIRPasses( codon::ir::transform::PassManager *pm, bool debug ) {
  // The order is MHEOptimizations, then ExpressivenessTransformations, then
  // `core-parallel-openmp`.
  //
  // Both passes realize brand-new function subtrees: ExpressivenessTransformations
  // when it rewrites secure operators (e.g. `ndarray @ Ciphertensor` ->
  // `secure_matmul(mpc, ...)`), MHEOptimizations for the operators and run-time
  // helpers it emits. Those subtrees can contain `@par` loops (CKKS/ring code).
  // Such loops must be lowered by `core-parallel-openmp`, otherwise the LLVM
  // visitor aborts with "parallel for-loop not lowered", so both passes run
  // BEFORE that pass in every pipeline. (Anchoring to "" appends at the end in
  // debug/JIT pipelines, after openmp, and `core-folding-pass-group:2` is itself
  // a folding group that runs after openmp lowering in release.)
  // `core-parallel-openmp` is present in every pipeline, so anchor to it.
  //
  // MHEOptimizations (@mhe_cipher_opt; @mhe_enc_opt is accepted and does
  // nothing) runs before the expressiveness pass: it plans on the operators as
  // the user wrote them (`__add__`, `__mul__`, `__matmul__`), and the
  // expressiveness pass then rewrites the operators it emitted like any other.
  pm->registerPass(std::make_unique<ExpressivenessTransformations>(), "core-parallel-openmp");
  pm->registerPass(std::make_unique<MHEOptimizations>(), "sequre-expressiveness-transformation");
}

} // namespace sequre

extern "C" std::unique_ptr<codon::DSL> load() { return std::make_unique<sequre::Sequre>(); }
