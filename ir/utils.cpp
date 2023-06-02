#include "codon/cir/util/irtools.h"

namespace sequre {

using namespace codon::ir;

const std::string sharedTensorTypeName = "SharedTensor";
const std::string cipherTensorTypeName = "CipherTensor";
const std::string MPPTypeName = "MPP";


bool isSequreFunc(Func *f) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.sequre");
}

bool isPolyOptFunc(Func *f) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.mpc_poly_opt");
}

bool isFactOptFunc(Func *f) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.mhe_mat_opt");
}

bool isCipherOptFunc( Func *f ) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.mhe_cipher_opt");
}

bool isSharedTensor( types::Type *t ) {
  return t->getName().find(sharedTensorTypeName) != std::string::npos;
}

bool isCipherTensor( types::Type *t ) {
  return t->getName().find(cipherTensorTypeName) != std::string::npos;
}

bool isMPP( types::Type *t ) {
  return t->getName().find(MPPTypeName) != std::string::npos;
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

void countVarUsage( Value *instruction,
                    std::set<codon::ir::id_t> &whitelist ) {

  for ( auto *usedValue : instruction->getUsedValues() ) {
    auto *var = util::getVar(usedValue);
    if ( var ) whitelist.insert(var->getId());
    else countVarUsage(usedValue, whitelist);
  }
  
}

void eliminateDeadAssignments(SeriesFlow *series, std::set<codon::ir::id_t> &whitelist) {
  auto it = series->begin();
  while ( it != series->end() ) {
    auto *assIns = cast<AssignInstr>(*it);
    if ( !assIns ) {
      ++it;
      continue;
    }

    auto *lhsVar = assIns->getLhs();
    if ( !whitelist.count(lhsVar->getId()) )
      it = series->erase(it);
    else ++it;
  }
}

void eliminateDeadCode(SeriesFlow *series) {
  std::set<codon::ir::id_t> whitelist;
  for ( auto it = series->begin(); it != series->end(); ++it ) countVarUsage(*it, whitelist);
  eliminateDeadAssignments(series, whitelist);
}

} // namespace sequre
