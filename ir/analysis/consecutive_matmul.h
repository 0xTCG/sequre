#pragma once

#include "codon/cir/util/irtools.h"

namespace sequre {

using namespace codon::ir;

void reorderConsecutiveMatmuls( SeriesFlow *, Value * );

} // namespace sequre
