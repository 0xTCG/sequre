#pragma once

#include "codon/cir/util/irtools.h"

namespace sequre {

using namespace codon::ir;

void countVarUsage( Value *, std::set<codon::ir::id_t> & );
void eliminateDeadAssignments( SeriesFlow *, std::set<codon::ir::id_t> & );
void eliminateDeadCode( SeriesFlow * );

} // namespace sequre
