#pragma once

namespace sequre {

using namespace codon::ir;

// IR internal
std::pair<std::vector<Value *>, std::vector<Type *>> getTypedArgs( CallInstr *, int skip = 0 );
bool isUnaryInstr( CallInstr * );
bool isBinaryInstr( CallInstr * );

// Attribute checks
bool hasSequreAttr( Func * );
bool hasPolyOptAttr( Func * );
bool hasCipherOptAttr( Func * );
bool hasEncOptAttr( Func * );
bool hasMatmulReorderOptAttr ( Func * ); 
bool hasDebugAttr( Func * );

// Secure types helpers
bool hasCKKSPlaintext( Type * );
bool hasCKKSCiphertext( Type * );

bool isCKKSPlaintext( Type * );
bool isCKKSCiphertext( Type * );
bool isSharetensor( Type * );
bool isCiphertensor( Type * );
bool isMPP( Type * );
bool isMPA( Type * );
bool isMPU( Type * );
bool isMP( Type * );
bool isSecureContainer( Type * );
bool isMPC( Value * );

Type *getTupleType( int, Type *, Module * );
Type *getTupleType( std::vector<Value *>, Module * );
Func        *getOrRealizeSequreInternalMethod(
                Module *, std::string const &,
                std::vector<Type *>,
                std::vector<Generic> );
Func        *getOrRealizeSequreOptimizationHelper(
                Module *M, std::string const &,
                std::vector<Type *>,
                std::vector<Generic> );

bool   isCallOfName( const Value *, const std::string & );
Value *findCallByName ( Value *, const std::string &, std::set<Value *> );
void   visitAllNodes( Value *, std::set<Value *> & );

// BET helpers
std::string const getOperation( CallInstr * );

// Secure calls
CallInstr *revealCall( Var *, VarValue * );

} // namespace sequre
