#include <array>

#include "codon/cir/util/irtools.h"
#include "codon/parser/common.h"
#include "utils.h"

namespace sequre {

using namespace codon::ir;

const std::string ckksPlaintextTypeSuffix = ".lattiseq.ckks.Ciphertext";
const std::string ckksCiphertextTypeSuffix = ".lattiseq.ckks.Plaintext";
const std::string sharetensorTypeSuffix = ".types.sharetensor.Sharetensor";
const std::string cipherTensorTypeSuffix = ".types.ciphertensor.Ciphertensor";
const std::string MPPTypeSuffix = ".types.multiparty_partition.MPP";
const std::string MPATypeSuffix = ".types.multiparty_aggregate.MPA";
const std::string MPUTypeSuffix = ".types.multiparty_union.MPU";

// Codon 0.19.6 names a plugin-provided module after the *first* registered
// search root that prefixes its file (codon/parser/common.cpp:get_root).
// The generic "<codon>/lib/codon/plugins" root (registered unconditionally
// for plugin auto-discovery) is an ancestor of the Sequre plugin's own
// stdlib root ("<codon>/lib/codon/plugins/sequre/stdlib", registered only
// once the plugin is loaded) and is checked first, so Sequre's modules are
// actually named "std.sequre.stdlib.sequre.*", not "std.sequre.*". Each
// lookup below tries both forms so this is robust to either resolution
// order without depending on codon-internal search-path precedence.
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

static bool hasTypeSuffix(types::Type *t, const std::string &suffix) {
  if (!t) return false;
  // Only match against the type's own base name, not nested generic
  // arguments (e.g. List[Ciphertensor[Ciphertext]] must not match
  // Ciphertensor, even though the substring appears nested inside it).
  std::string name = t->getName();
  auto bracketPos = name.find('[');
  std::string base = bracketPos == std::string::npos ? name : name.substr(0, bracketPos);
  return base.find(suffix) != std::string::npos;
}

static bool hasAttributeInAnyModule(Func *f, const std::array<const char *, 2> &modules,
                                    const char *name) {
  if (!f)
    return false;

  for (const auto *module : modules) {
    if (util::hasAttribute(f, codon::ast::getMangledFunc(module, name)))
      return true;
  }

  return false;
}

static types::Type *getOrRealizeAnyType(Module *M, const std::array<const char *, 2> &modules,
                                        const char *name,
                                        const std::vector<types::Generic> &generics) {
  for (const auto *module : modules) {
    if (auto *type = M->getOrRealizeType(codon::ast::getMangledClass(module, name), generics))
      return type;
  }

  return nullptr;
}


std::pair<std::vector<Value *>, std::vector<types::Type *>> getTypedArgs( CallInstr *v, int skip) {
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

bool isUnaryInstr(CallInstr *instr) {
    return instr->numArgs() == 1;
}

bool isBinaryInstr(CallInstr *instr) {
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

bool hasMatmulReorderOptAttr( Func *f ) {
  return hasAttributeInAnyModule(f, sequreAttributeModules, "reorder_matmul");
}

bool hasCipherOptAttr( Func *f ) {
  return hasAttributeInAnyModule(f, sequreAttributeModules, "mhe_cipher_opt");
}

bool hasEncOptAttr( Func *f ) {
  return hasAttributeInAnyModule(f, sequreAttributeModules, "mhe_enc_opt");
}

bool hasDebugAttr( Func *f ) {
  return hasAttributeInAnyModule(f, sequreAttributeModules, "debug");
}

bool hasCKKSPlaintext( types::Type *t ) {
  return hasTypeSuffix(t, ckksPlaintextTypeSuffix);
}

bool hasCKKSCiphertext( types::Type *t ) {
  return hasTypeSuffix(t, ckksCiphertextTypeSuffix);
}

bool isCKKSPlaintext( types::Type *t ) {
  return hasTypeSuffix(t, ckksPlaintextTypeSuffix);
}

bool isCKKSCiphertext( types::Type *t ) {
  return hasTypeSuffix(t, ckksCiphertextTypeSuffix);
}

bool isSharetensor( types::Type *t ) {
  return hasTypeSuffix(t, sharetensorTypeSuffix);
}

bool isCiphertensor( types::Type *t ) {
  return hasTypeSuffix(t, cipherTensorTypeSuffix);
}

bool isMPP( types::Type *t ) {
  return hasTypeSuffix(t, MPPTypeSuffix);
}

bool isMPA( types::Type *t ) {
  return hasTypeSuffix(t, MPATypeSuffix);
}

bool isMPU( types::Type *t ) {
  return hasTypeSuffix(t, MPUTypeSuffix);
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

Func *getOrRealizeSequreOptimizationHelper( Module *M, std::string const &funcName,
                                            std::vector<types::Type *> args,
                                            std::vector<types::Generic> generics ) {
  Func *func = nullptr;
  for (const auto *module : sequreOptimizationModules) {
    func = M->getOrRealizeFunc(funcName, args, generics, module);
    if (func)
      break;
  }

  if ( !func ) {
    std::cout << "\nSEQURE TYPE REALIZATION ERROR: Could not realize helper func: " << funcName
              << "\n\tfor parameters ";

    for ( auto arg : args )
      std::cout << "\n\t\t" << arg->getName();

    std::cout << std::endl;
  }

  return func;
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

Value *findCallByName( Value *value, const std::string &name, std::set<Value *> visited = {} ) {
  if ( visited.count(value) ) return nullptr;
  if ( isCallOfName(value, name) ) return value;

  for ( auto *usedValue : value->getUsedValues() )
    if ( auto *foundCall = findCallByName(usedValue, name, visited) )
      return foundCall;

  return nullptr;
}

void visitAllNodes( Value *value, std::set<Value *> &visited ) {
  visited.insert(value);
  for ( auto *usedValue : value->getUsedValues() ) visitAllNodes(usedValue, visited);
}

std::string const getOperation( CallInstr *callInstr ) {
  auto *callee = callInstr->getCallee();
  assert(callee);

  auto *func = util::getFunc(callee);
  assert(func);

  return func->getUnmangledName();
}

CallInstr *revealCall( Var *var, VarValue *mpc ) {
  assert( isSecureContainer(var->getType()) && "ERROR: Reveal call called on top of non-secure container" );
  auto *varType = var->getType();

  std::string namePath;
  if ( isSharetensor(varType) )
    namePath = sharetensorTypeSuffix;
  else if ( isCiphertensor(varType) )
    namePath = cipherTensorTypeSuffix;
  else if ( isMPP(varType) )
    namePath = MPPTypeSuffix;
  else if ( isMPA(varType) )
    namePath = MPATypeSuffix;
  else if ( isMPU(varType) )
    namePath = MPUTypeSuffix;
  else
    throw "ERROR: Reveal call called on top of non-secure container";

  auto *M          = var->getModule();
  auto *method     = M->getOrRealizeMethod(varType, "reveal", { varType, mpc->getType() }, {});
  if ( !method )
    std::cout << "\nSEQURE TYPE REALIZATION ERROR: Could not realize reveal method for " << varType->getName() << "\n";
  return util::call(method, { M->Nr<VarValue>(var), mpc });
}

} // namespace sequre
