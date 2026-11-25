#pragma once

#include <string>
#include <vector>

#include "codon/cir/cir.h"

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
bool hasDebugAttr( Func * );

// Secure types helpers. A class is recognized by its dotted path at a component boundary at the end of the type's
// base name (the name before its generic arguments), so both install layouts match (std.sequre.types.* when the
// plugin's stdlib is a symlink, std.sequre.stdlib.sequre.types.* when it is a copy) and no other class does
// (MPPAligner is not an MPP).
bool isSharetensor( types::Type * );
bool isCiphertensor( types::Type * );
bool isMPP( types::Type * );
bool isMPA( types::Type * );
bool isMPU( types::Type * );
bool isMP( types::Type * );
bool isSecureContainer( types::Type * );
bool isMPC( Value * );

// The operand kinds the IR optimizations reason about, decided from the realized type and its generic arguments.
enum class OperandKind { UNKNOWN, INT, FLOAT, NDARRAY, SHARETENSOR, CIPHER_CT, PLAIN_CT, MPP, MPA, MPU };
OperandKind operandKind( types::Type * );
// Values whose operations involve the secure protocols: Sharetensor, Ciphertensor[Ciphertext], MPP, MPA, MPU
bool isSecretKind( OperandKind );
bool isScalarKind( OperandKind );
bool isTensorKind( OperandKind );
const char *operandKindName( OperandKind );
// A public operand whose arithmetic rounds rather than wraps: a float, or an ndarray of float or float32 (an int, or an
// ndarray of ints, wraps at 64 bits where Sequre's arithmetic on a secure operand does not)
bool isFloatPublic( types::Type * );
// MPCEnv[...], without asserting on other types
bool isMPCType( types::Type * );

types::Type *getTupleType( int, types::Type *, Module * );
types::Type *getTupleType( std::vector<Value *>, Module * );
Func        *getOrRealizeSequreInternalMethod(
                Module *, std::string const &,
                std::vector<types::Type *>,
                std::vector<types::Generic> );

// Realization that reports nothing on failure (the IR optimizations try alternatives and leave code as written when
// none realizes). Each returns nullptr unless the realized function's parameter types are exactly the given ones.
Func *realizeMethodQuiet( Module *, types::Type *parent, const std::string &name,
                          const std::vector<types::Type *> &args );
Func *realizeInternalMethodQuiet( Module *, const std::string &name, const std::vector<types::Type *> &args );
Func *realizeOptimizationHelperQuiet( Module *, const std::string &name, const std::vector<types::Type *> &args );
// True if the function's parameter types are exactly the given ones
bool hasParamTypes( Func *, const std::vector<types::Type *> & );
// The function's return type, or nullptr
types::Type *returnTypeOf( Func * );

bool   isCallOfName( const Value *, const std::string & );

// Diagnostics of the IR optimizations: with SEQURE_IR_OPT_DEBUG set (and not "0") they print what they rewrite and
// what they leave as written to stderr
bool irOptDebug();
// A broken invariant of the IR optimizations (not anything a user wrote): a compile error naming the function
void irOptInternalError( Func *, const std::string &what );

// Secure calls
CallInstr *revealCall( Var *, VarValue * );

} // namespace sequre
