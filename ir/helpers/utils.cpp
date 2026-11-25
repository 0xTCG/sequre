#include <array>
#include <cstdlib>
#include <iostream>

#include "codon/cir/util/irtools.h"
#include "codon/parser/common.h"
#include "codon/util/common.h"
#include "utils.h"

namespace sequre {

using namespace codon::ir;

// Dotted class paths below the install-dependent module prefix (see hasClassPath)
const std::string ckksCiphertextClassPath = "sequre.lattiseq.ckks.Ciphertext";
const std::string ckksPlaintextClassPath = "sequre.lattiseq.ckks.Plaintext";
const std::string sharetensorClassPath = "sequre.types.sharetensor.Sharetensor";
const std::string ciphertensorClassPath = "sequre.types.ciphertensor.Ciphertensor";
const std::string MPPClassPath = "sequre.types.multiparty_partition.MPP";
const std::string MPAClassPath = "sequre.types.multiparty_aggregate.MPA";
const std::string MPUClassPath = "sequre.types.multiparty_union.MPU";
const std::string ndarrayClassPath = "numpy.ndarray.ndarray";
const std::string MPCEnvClassPath = "sequre.mpc.env.MPCEnv";

// Codon 0.19.6 names a plugin-provided module after the *first* registered
// search root that prefixes its file (codon/parser/common.cpp:get_root).
// The generic "<codon>/lib/codon/plugins" root (registered unconditionally
// for plugin auto-discovery) is an ancestor of the Sequre plugin's own
// stdlib root ("<codon>/lib/codon/plugins/sequre/stdlib", registered only
// once the plugin is loaded) and is checked first, so in an install whose
// plugin stdlib is a real directory Sequre's modules are named
// "std.sequre.stdlib.sequre.*"; when the plugin's stdlib is a symlink to a
// checkout (a development install) they are named "std.sequre.*". Each
// lookup below tries both forms.
static constexpr std::array<const char *, 2> sequreAttributeModules = {
  "std.sequre.attributes",
  "std.sequre.stdlib.sequre.attributes",
};

static constexpr std::array<const char *, 2> sequreRuntimeModules = {
  "std.sequre.runtime",
  "std.sequre.stdlib.sequre.runtime",
};

static constexpr std::array<const char *, 2> sequreMpcEnvModules = {
  "std.sequre.mpc.env",
  "std.sequre.stdlib.sequre.mpc.env",
};

static constexpr std::array<const char *, 2> sequreInternalModules = {
  "std.sequre.types.internal",
  "std.sequre.stdlib.sequre.types.internal",
};

static constexpr std::array<const char *, 2> sequreOptimizationModules = {
  "std.optimization.ir.__init__",
  "std.sequre.stdlib.optimization.ir.__init__",
};

// The type's own class (its name up to the generic arguments, without the numeric component Codon appends to a
// realized class: std.sequre.types.sharetensor.Sharetensor.0[UInt[192]]) is <prefix>.<path> for some module prefix:
// the path matches whole dotted components at the end of the name, so List[Ciphertensor[...]] is no Ciphertensor and
// MPPAligner no MPP.
static bool hasClassPath( types::Type *t, const std::string &path ) {
  if ( !t ) return false;
  std::string name = t->getName();
  auto bracketPos = name.find('[');
  std::string base = bracketPos == std::string::npos ? name : name.substr(0, bracketPos);
  auto dotPos = base.rfind('.');
  if ( dotPos != std::string::npos && dotPos + 1 < base.size() &&
       base.find_first_not_of("0123456789", dotPos + 1) == std::string::npos )
    base = base.substr(0, dotPos);
  if ( base.size() <= path.size() ) return false;
  return base.compare(base.size() - path.size(), path.size(), path) == 0 && base[base.size() - path.size() - 1] == '.';
}

static bool hasAttributeInAnyModule( Func *f, const std::array<const char *, 2> &modules, const char *name ) {
  if ( !f ) return false;

  for ( const auto *module : modules ) {
    if ( util::hasAttribute(f, codon::ast::getMangledFunc(module, name)) )
      return true;
  }

  return false;
}

static types::Type *getOrRealizeAnyType( Module *M, const std::array<const char *, 2> &modules,
                                         const char *name,
                                         const std::vector<types::Generic> &generics ) {
  for ( const auto *module : modules ) {
    if ( auto *type = M->getOrRealizeType(codon::ast::getMangledClass(module, name), generics) )
      return type;
  }

  return nullptr;
}


std::pair<std::vector<Value *>, std::vector<types::Type *>> getTypedArgs( CallInstr *v, int skip ) {
    std::vector<Value *> args;
    std::vector<types::Type *> types;

    int idx = 0;
    for ( auto it = v->begin(); it != v->end(); ++it, ++idx ) {
      if ( idx < skip )
        continue;

      auto *arg = *it;
      args.push_back(arg);
      types.push_back(arg->getType());
    }

    return std::make_pair(args, types);
}

bool isUnaryInstr( CallInstr *instr ) {
    return instr->numArgs() == 1;
}

bool isBinaryInstr( CallInstr *instr ) {
    return instr->numArgs() == 2;
}

bool hasSequreAttr( Func *f ) {
  return hasAttributeInAnyModule(f, sequreAttributeModules, "sequre") ||
         hasAttributeInAnyModule(f, sequreRuntimeModules, "local") ||
         hasAttributeInAnyModule(f, sequreRuntimeModules, "online") ||
         hasAttributeInAnyModule(f, sequreRuntimeModules, "main");
}

bool hasPolyOptAttr( Func *f ) {
  return hasAttributeInAnyModule(f, sequreAttributeModules, "mpc_poly_opt");
}

bool hasCipherOptAttr( Func *f ) {
  return hasAttributeInAnyModule(f, sequreAttributeModules, "mhe_cipher_opt");
}

bool hasDebugAttr( Func *f ) {
  return hasAttributeInAnyModule(f, sequreAttributeModules, "debug");
}

bool isSharetensor( types::Type *t ) {
  return hasClassPath(t, sharetensorClassPath);
}

bool isCiphertensor( types::Type *t ) {
  return hasClassPath(t, ciphertensorClassPath);
}

bool isMPP( types::Type *t ) {
  return hasClassPath(t, MPPClassPath);
}

bool isMPA( types::Type *t ) {
  return hasClassPath(t, MPAClassPath);
}

bool isMPU( types::Type *t ) {
  return hasClassPath(t, MPUClassPath);
}

bool isMP( types::Type *t ) {
  return isMPP(t) || isMPA(t) || isMPU(t);
}

bool isSecureContainer( types::Type *t ) {
  return isSharetensor(t) || isCiphertensor(t) || isMP(t);
}

bool isMPC( Value *value ) {
  auto generics = value->getType()->getGenerics();
  assert( generics.size() == 1 && "ERROR: While testing if value is the MPC instance. It should have one and only one generic type." );
  auto *M = value->getModule();
  auto *mpcType = getOrRealizeAnyType(M, sequreMpcEnvModules, "MPCEnv", generics);
  assert(mpcType);
  return value->getType()->is(mpcType);
}

bool isMPCType( types::Type *t ) {
  return t && hasClassPath(t, MPCEnvClassPath);
}

OperandKind operandKind( types::Type *t ) {
  if ( !t ) return OperandKind::UNKNOWN;
  auto *M = t->getModule();
  if ( M ) {
    if ( t->is(M->getIntType()) ) return OperandKind::INT;
    if ( t->is(M->getFloatType()) ) return OperandKind::FLOAT;
  }
  if ( hasClassPath(t, ndarrayClassPath) ) return OperandKind::NDARRAY;
  if ( hasClassPath(t, sharetensorClassPath) ) return OperandKind::SHARETENSOR;
  if ( hasClassPath(t, MPPClassPath) ) return OperandKind::MPP;
  if ( hasClassPath(t, MPAClassPath) ) return OperandKind::MPA;
  if ( hasClassPath(t, MPUClassPath) ) return OperandKind::MPU;
  if ( hasClassPath(t, ciphertensorClassPath) ) {
    // Ciphertensor[ctype]: what it holds is its generic argument
    auto generics = t->getGenerics();
    if ( generics.size() != 1 || !generics[0].isType() ) return OperandKind::UNKNOWN;
    auto *ctype = generics[0].getTypeValue();
    if ( hasClassPath(ctype, ckksCiphertextClassPath) ) return OperandKind::CIPHER_CT;
    if ( hasClassPath(ctype, ckksPlaintextClassPath) ) return OperandKind::PLAIN_CT;
    return OperandKind::UNKNOWN;
  }
  return OperandKind::UNKNOWN;
}

bool isFloatPublic( types::Type *t ) {
  auto k = operandKind(t);
  if ( k == OperandKind::FLOAT ) return true;
  if ( k != OperandKind::NDARRAY ) return false;
  // ndarray[S, T]: the element type is the second generic argument
  auto generics = t->getGenerics();
  auto *M = t->getModule();
  if ( !M || generics.size() != 2 || !generics[1].isType() ) return false;
  auto *dtype = generics[1].getTypeValue();
  return dtype && (dtype->is(M->getFloatType()) || dtype->is(M->getFloat32Type()));
}

bool isSecretKind( OperandKind k ) {
  return k == OperandKind::SHARETENSOR || k == OperandKind::CIPHER_CT || k == OperandKind::MPP ||
         k == OperandKind::MPA || k == OperandKind::MPU;
}

bool isScalarKind( OperandKind k ) {
  return k == OperandKind::INT || k == OperandKind::FLOAT;
}

bool isTensorKind( OperandKind k ) {
  return k != OperandKind::UNKNOWN && !isScalarKind(k);
}

const char *operandKindName( OperandKind k ) {
  switch ( k ) {
  case OperandKind::INT: return "int";
  case OperandKind::FLOAT: return "float";
  case OperandKind::NDARRAY: return "ndarray";
  case OperandKind::SHARETENSOR: return "Sharetensor";
  case OperandKind::CIPHER_CT: return "Ciphertensor[Ciphertext]";
  case OperandKind::PLAIN_CT: return "Ciphertensor[Plaintext]";
  case OperandKind::MPP: return "MPP";
  case OperandKind::MPA: return "MPA";
  case OperandKind::MPU: return "MPU";
  default: return "unknown";
  }
}

types::Type *getTupleType( int n, types::Type *elemType, Module *M ) {
  std::vector<types::Type *> tupleTypes;
  for (int i = 0; i != n; ++i) tupleTypes.push_back(elemType);
  return M->getTupleType(tupleTypes);
}

types::Type *getTupleType( std::vector<Value *> vals, Module *M ) {
  std::vector<types::Type *> tupleTypes;
  for ( auto *v : vals ) tupleTypes.push_back(v->getType());
  return M->getTupleType(tupleTypes);
}

Func *getOrRealizeSequreInternalMethod( Module *M, std::string const &methodName,
                                        std::vector<types::Type *> args,
                                        std::vector<types::Generic> generics ) {
  auto *sequreInternalType = getOrRealizeAnyType(M, sequreInternalModules, "Internal", {});
  auto *method = M->getOrRealizeMethod(sequreInternalType, methodName, args, generics);

  if ( !method ) {
    std::cout << "\nSEQURE TYPE REALIZATION ERROR: Could not realize internal method: " << methodName
              << "\n\tfor parameters ";

    for ( auto arg : args )
      std::cout << "\n\t\t" << arg->getName();

    std::cout << std::endl;
  }

  return method;
}

bool hasParamTypes( Func *f, const std::vector<types::Type *> &args ) {
  if ( !f ) return false;
  auto *ft = cast<types::FuncType>(f->getType());
  if ( !ft ) return false;
  std::vector<types::Type *> params(ft->begin(), ft->end());
  if ( params.size() != args.size() ) return false;
  for ( size_t i = 0; i < params.size(); ++i )
    if ( !params[i] || !args[i] || !params[i]->is(args[i]) ) return false;
  return true;
}

types::Type *returnTypeOf( Func *f ) {
  if ( !f ) return nullptr;
  auto *ft = cast<types::FuncType>(f->getType());
  return ft ? ft->getReturnType() : nullptr;
}

Func *realizeMethodQuiet( Module *M, types::Type *parent, const std::string &name,
                          const std::vector<types::Type *> &args ) {
  if ( !M || !parent || !parent->getAstType() ) return nullptr;
  for ( auto *a : args )
    if ( !a || !a->getAstType() ) return nullptr;
  Func *f = nullptr;
  try {
    f = M->getOrRealizeMethod(parent, name, args);
  } catch ( ... ) {
    return nullptr;
  }
  return hasParamTypes(f, args) ? f : nullptr;
}

Func *realizeInternalMethodQuiet( Module *M, const std::string &name, const std::vector<types::Type *> &args ) {
  types::Type *internalType = nullptr;
  try {
    internalType = getOrRealizeAnyType(M, sequreInternalModules, "Internal", {});
  } catch ( ... ) {
    return nullptr;
  }
  return internalType ? realizeMethodQuiet(M, internalType, name, args) : nullptr;
}

Func *realizeOptimizationHelperQuiet( Module *M, const std::string &name, const std::vector<types::Type *> &args ) {
  for ( auto *a : args )
    if ( !a || !a->getAstType() ) return nullptr;
  for ( const auto *module : sequreOptimizationModules ) {
    Func *f = nullptr;
    try {
      f = M->getOrRealizeFunc(name, args, {}, module);
    } catch ( ... ) {
      f = nullptr;
    }
    if ( f ) return hasParamTypes(f, args) ? f : nullptr;
  }
  return nullptr;
}

bool isCallOfName( const Value *value, const std::string &name ) {
  if (auto *call = cast<CallInstr>(value)) {
    auto *fn = util::getFunc(call->getCallee());
    if ( !fn || call->numArgs() == 0 || fn->getUnmangledName() != name )
      return false;

    return true;
  }

  return false;
}

bool irOptDebug() {
  static const bool on = [] {
    const char *e = std::getenv("SEQURE_IR_OPT_DEBUG");
    return e && *e && std::string(e) != "0";
  }();
  return on;
}

void irOptInternalError( Func *f, const std::string &what ) {
  auto info = f ? f->getSrcInfo() : codon::SrcInfo();
  codon::compilationError("Sequre IR optimization internal error in function '" +
                              (f ? f->getUnmangledName() : std::string("?")) + "': " + what,
                          info.file, info.line, info.col);
  std::exit(1);  // compilationError terminates; never reached
}

CallInstr *revealCall( Var *var, VarValue *mpc ) {
  assert( isSecureContainer(var->getType()) && "ERROR: Reveal call called on top of non-secure container" );
  auto *varType = var->getType();
  auto *M          = var->getModule();
  auto *method     = M->getOrRealizeMethod(varType, "reveal", { varType, mpc->getType() }, {});
  if ( !method )
    std::cout << "\nSEQURE TYPE REALIZATION ERROR: Could not realize reveal method for " << varType->getName() << "\n";
  return util::call(method, { M->Nr<VarValue>(var), mpc });
}

} // namespace sequre
