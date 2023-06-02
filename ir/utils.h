#pragma once

namespace sequre {

using namespace codon::ir;

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

} // namespace sequre
