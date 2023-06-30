#pragma once

#include "enums.h"

namespace sequre {

using namespace codon::ir;

// Secure types helpers
bool isSequreFunc( Func * );
bool isPolyOptFunc( Func * );
bool isCipherOptFunc( Func * );
bool isMatmulReorderOptFunc ( Func * ); 

bool hasCKKSPlaintext( types::Type * );
bool hasCKKSCiphertext( types::Type * );

bool isCKKSPlaintext( types::Type * );
bool isCKKSCiphertext( types::Type * );
bool isSharedTensor( types::Type * );
bool isCipherTensor( types::Type * );
bool isMPP( types::Type * );
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

} // namespace sequre
