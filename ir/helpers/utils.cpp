#include "codon/cir/util/irtools.h"
#include "utils.h"

namespace sequre {

using namespace codon::ir;

const std::string ckksPlaintextTypeName = "std.sequre.lattiseq.ckks.Ciphertext";
const std::string ckksCiphertextTypeName = "std.sequre.lattiseq.ckks.Plaintext";
const std::string sharetensorTypeName = "std.sequre.types.sharetensor.Sharetensor";
const std::string cipherTensorTypeName = "std.sequre.types.ciphertensor.Ciphertensor";
const std::string MPPTypeName = "std.sequre.types.multiparty_partition.MPP";
const std::string MPATypeName = "std.sequre.types.multiparty_aggregate.MPA";
const std::string MPUTypeName = "std.sequre.types.multiparty_union.MPU";


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
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.sequre");
}

bool hasPolyOptAttr( Func *f ) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.mpc_poly_opt");
}

bool hasMatmulReorderOptAttr( Func *f ) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.reorder_matmul");
}

bool hasCipherOptAttr( Func *f ) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.mhe_cipher_opt");
}

bool hasEncOptAttr( Func *f ) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.mhe_enc_opt");
}

bool hasDebugAttr( Func *f ) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.debug");
}

bool hasCKKSPlaintext( types::Type *t ) {
  return t->getName().find(ckksPlaintextTypeName) != std::string::npos;
}

bool hasCKKSCiphertext( types::Type *t ) {
  return t->getName().find(ckksCiphertextTypeName) != std::string::npos;
}

bool isCKKSPlaintext( types::Type *t ) {
  return t->getName().rfind(ckksPlaintextTypeName, 0) == 0;
}

bool isCKKSCiphertext( types::Type *t ) {
  return t->getName().rfind(ckksCiphertextTypeName, 0) == 0;
}

bool isSharetensor( types::Type *t ) {
  return t->getName().rfind(sharetensorTypeName, 0) == 0;
}

bool isCiphertensor( types::Type *t ) {
  return t->getName().rfind(cipherTensorTypeName, 0) == 0;
}

bool isMPP( types::Type *t ) {
  return t->getName().rfind(MPPTypeName, 0) == 0;
}

bool isMPA( types::Type *t ) {
  return t->getName().rfind(MPATypeName, 0) == 0;
}

bool isMPU( types::Type *t ) {
  return t->getName().rfind(MPUTypeName, 0) == 0;
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
  auto *mpcType = M->getOrRealizeType("MPCEnv", { generics[0] }, "std.sequre.mpc.env");
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
  auto *sequreInternalType = M->getOrRealizeType("Internal", {}, "std.sequre.types.internal");
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
  auto *func = M->getOrRealizeFunc(funcName, args, generics, "std.optimization.ir.__init__");
  
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
    namePath = sharetensorTypeName;
  else if ( isCiphertensor(varType) )
    namePath = cipherTensorTypeName;
  else if ( isMPP(varType) )
    namePath = MPPTypeName;
  else if ( isMPA(varType) )
    namePath = MPATypeName;
  else if ( isMPU(varType) )
    namePath = MPUTypeName;
  else
    throw "ERROR: Reveal call called on top of non-secure container";
  
  auto *M          = var->getModule();
  auto *method     = M->getOrRealizeMethod(varType, "reveal", { varType, mpc->getType() }, {});
  if ( !method )
    std::cout << "\nSEQURE TYPE REALIZATION ERROR: Could not realize reveal method for " << varType->getName() << "\n";
  return util::call(method, { M->Nr<VarValue>(var), mpc });
}

} // namespace sequre
