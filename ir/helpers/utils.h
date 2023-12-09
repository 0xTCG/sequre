#pragma once

#include "enums.h"

namespace sequre {

using namespace codon::ir;

// Attribute checks
bool hasSequreAttr( Func * );
bool hasPolyOptAttr( Func * );
bool hasCipherOptAttr( Func * );
bool hasMatmulReorderOptAttr ( Func * ); 
bool hasDebugAttr( Func * );

// Secure types helpers
bool hasCKKSPlaintext( types::Type * );
bool hasCKKSCiphertext( types::Type * );

bool isCKKSPlaintext( types::Type * );
bool isCKKSCiphertext( types::Type * );
bool isSharetensor( types::Type * );
bool isCiphertensor( types::Type * );
bool isMPP( types::Type * );
bool isMPA( types::Type * );
bool isMPU( types::Type * );
bool isMP( types::Type * );
bool isSecureContainer( types::Type * );
bool isMPC( Value * );

types::Type *getTupleType( int, types::Type *, Module * );
types::Type *getTupleType( std::vector<Value *>, Module * );
Func        *getOrRealizeSequreInternalMethod( Module *, std::string const &,
                                        std::vector<types::Type *>,
                                        std::vector<types::Generic> );
Func        *getOrRealizeSequreHelper( Module *M, std::string const &,
                                std::vector<types::Type *>,
                                std::vector<types::Generic> );

bool   isCallOfName( const Value *, const std::string & );
Value *findCallByName ( Value *, const std::string &, std::set<Value *> = {} );
void   visitAllNodes( Value *, std::set<Value *> & );

// BET helpers
bool isArithmeticOperation( Operation );
Operation getOperation( CallInstr * );

// Secure calls
CallInstr *revealCall( Var *, VarValue * );

} // namespace sequre
