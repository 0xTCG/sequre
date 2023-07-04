#include "enums.h"
#include "codon/cir/util/irtools.h"

namespace sequre {

using namespace codon::ir;

const std::string ckksPlaintextTypeName = "std.sequre.lattiseq.ckks.Ciphertext";
const std::string ckksCiphertextTypeName = "std.sequre.lattiseq.ckks.Plaintext";
const std::string sharetensorTypeName = "std.sequre.types.sharetensor.Sharetensor";
const std::string cipherTensorTypeName = "std.sequre.types.ciphertensor.Ciphertensor";
const std::string MPPTypeName = "std.sequre.types.multiparty_partition.MPP";


bool isSequreFunc( Func *f ) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.sequre");
}

bool isPolyOptFunc( Func *f ) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.mpc_poly_opt");
}

bool isMatmulReorderOptFunc( Func *f ) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.reorder_matmul");
}

bool isCipherOptFunc( Func *f ) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.mhe_cipher_opt");
}

bool hasCKKSPlaintext( types::Type *t ) {
  return t->getName().find(ckksPlaintextTypeName) != std::string::npos;
}

bool hasCKKSCiphertext( types::Type *t ) {
  return t->getName().find(ckksCiphertextTypeName) != std::string::npos;
}

bool isCKKSPlaintext( types::Type *t ) {
  return t->getName().rfind(ckksPlaintextTypeName, 0) != std::string::npos;
}

bool isCKKSCiphertext( types::Type *t ) {
  return t->getName().rfind(ckksCiphertextTypeName, 0) != std::string::npos;
}

bool isSharetensor( types::Type *t ) {
  return t->getName().rfind(sharetensorTypeName, 0) != std::string::npos;
}

bool isCiphertensor( types::Type *t ) {
  return t->getName().rfind(cipherTensorTypeName, 0) != std::string::npos;
}

bool isMPP( types::Type *t ) {
  return t->getName().rfind(MPPTypeName, 0) != std::string::npos;
}

bool isSecureContainer( types::Type *t ) {
  return isSharetensor(t) || isCiphertensor(t) || isMPP(t);
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
    
    for (auto arg : args)
      std::cout << "\n\t\t" << arg->getName();
              
    std::cout << std::endl;
  }
  
  return method;
}

Func *getOrRealizeSequreHelper( Module *M, std::string const &funcName,
                                std::vector<types::Type *> args,
                                std::vector<types::Generic> generics ) {
  auto *func = M->getOrRealizeFunc(funcName, args, generics, "std.helpers");
  
  if ( !func ) {
    std::cout << "\nSEQURE TYPE REALIZATION ERROR: Could not realize helper func: " << funcName
              << "\n\tfor parameters ";
    
    for (auto arg : args)
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

Value *findCallByName ( Value *value, const std::string &name, std::set<Value *> visited = {} ) {
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

bool isArithmeticOperation( Operation op ) { return op == add || op == mul || op == matmul || op == power; }

Operation getOperation( CallInstr *callInstr ) {
  auto *f        = util::getFunc(callInstr->getCallee());
  auto instrName = f->getUnmangledName();
  
  if ( instrName == Module::ADD_MAGIC_NAME ) return add;
  if ( instrName == Module::MUL_MAGIC_NAME ) return mul;
  if ( instrName == Module::POW_MAGIC_NAME ) return power;
  if ( instrName == Module::MATMUL_MAGIC_NAME ) return matmul;
  
  return noop;
}

} // namespace sequre
