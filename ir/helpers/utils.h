#pragma once

namespace sequre {

using namespace codon::ir;

// IR internal
std::pair<std::vector<Value *>, std::vector<types::Type *>> getTypedArgs( CallInstr *, int skip = 0 );
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
Func        *getOrRealizeSequreInternalMethod(
                Module *, std::string const &,
                std::vector<types::Type *>,
                std::vector<types::Generic> );
Func        *getOrRealizeSequreOptimizationHelper(
                Module *M, std::string const &,
                std::vector<types::Type *>,
                std::vector<types::Generic> );

bool   isCallOfName( const Value *, const std::string & );
Value *findCallByName ( Value *, const std::string &, std::set<Value *> );
void   visitAllNodes( Value *, std::set<Value *> & );

// BET helpers
std::string const getOperation( CallInstr * );

// Secure calls
CallInstr *revealCall( Var *, VarValue * );

} // namespace sequre
