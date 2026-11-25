#pragma once

#include "codon/cir/cir.h"

namespace sequre {

/// @mhe_cipher_opt on one realized function. It factors common operands out of sums of products, adds and
/// multiplies public operands among themselves first, folds a definition into its only use when nothing can tell,
/// and hands chains of `@` to the run-time ordering helper. Every rewrite is planned on a representation of its own,
/// typed by Codon and checked against the cost of the code as written before anything in the function changes:
/// what cannot be planned, typed or proven cheaper stays as written.
void applyCipherOpt( codon::ir::BodiedFunc *f );

} // namespace sequre
