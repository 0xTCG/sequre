#pragma once

#include "enums.h"

namespace sequre {

using namespace codon::ir;

// Secure types helpers
bool isSequreFunc( Func * );
bool isPolyOptFunc( Func * );
bool isFactOptFunc( Func * );
bool isCipherOptFunc( Func * );
bool isSharedTensor( types::Type * );
bool isCipherTensor( types::Type * );
bool isMPP( types::Type * );

Func *getOrRealizeSequreInternalMethod( Module *, std::string const &,
                                        std::vector<types::Type *>,
                                        std::vector<types::Generic>);

// Dead code elimination
void countVarUsage( Value *, std::set<codon::ir::id_t> &, std::set<codon::ir::id_t> & );
void eliminateDeadAssignments( SeriesFlow *, std::set<codon::ir::id_t> & );
void eliminateDeadCode( SeriesFlow * );

// BET helpers
bool isArithmeticOperation( Operation );
Operation getOperation( CallInstr * );

} // namespace sequre
